import { z } from "zod";
import type { GraphAccess } from "./access.js";
import { ExtractionError, StoreError, TransientError } from "./errors.js";

import { dumps, loads } from "./internal/json.js";
import { contextPayload } from "./internal/payload.js";
import { comparePy, pyStr } from "./internal/py.js";
import { L_ENTITY } from "./labels.js";
import { summarizeError } from "./llm/index.js";
import { type InsertWarning, insertWarning, type JsonObject, type Relation } from "./models.js";
import * as ops from "./ops.js";
import type { EdgeRow, GraphPrimitives, LLMClient, Props } from "./protocols.js";
import { decodeEntity } from "./serde.js";

const MAX_RIVALS = 20;
export const PENDING = "judge_pending_json";
const SYSTEM =
  "You decide whether a new fact makes older facts about the same subject no longer true. List an older fact " +
  "only when it and the new fact cannot both hold at once, such as a previous employer after a job change or " +
  "a liking that became a dislike. Facts that can hold together, such as knowing several people, are not " +
  "contradicted. When unsure, list nothing.";

const Verdict = z.object({ retired: z.array(z.number().int()).default([]) });

/**
 * Render one fact for the judge.
 *
 * @param edgeType (string): the relation type.
 * @param source (string): the source entity's name.
 * @param target (string): the destination entity's name.
 * @param properties (JsonObject): the relation's properties.
 * @return (string): the fact as one line, with line breaks and control characters escaped.
 */
function line(edgeType: string, source: string, target: string, properties: JsonObject): string {
  const entries = Object.entries(properties).sort(([a], [b]) => comparePy(a, b));
  const props = entries.length ? ` {${entries.map(([k, v]) => `${k}=${pyStr(v)}`).join(", ")}}` : "";
  return contextPayload(`${source} -[${edgeType}]-> ${target}${props}`);
}

/** A relation a committed write left for the judge, with what its write recorded beside it. */
export interface Pending {
  readonly relation: Relation;
  readonly batch: readonly string[];
  readonly source_id: string | null;
  readonly source_version: string | null;
  readonly deferred: readonly string[];
}

/**
 * Encode a pending judgment as the marker stored on its relation version.
 *
 * @param item (Pending): the pending judgment.
 * @return (string): the marker JSON.
 */
function encode(item: Pending): string {
  return dumps({
    batch: item.batch,
    source_id: item.source_id,
    source_version: item.source_version,
    deferred: item.deferred,
  });
}

/**
 * Mark each relation a write created or revised that has a rival as pending judgment, inside that write.
 *
 * @param runner (GraphPrimitives): the graph runner, inside the write.
 * @param ns (string): the namespace.
 * @param written (readonly Relation[]): the relations the write stored.
 * @param caps (ReadonlyMap<string, ops.RelationCap> | null): the schema's max_count limits by relation type.
 * @param writtenAt (Date): the write's timestamp; only relations it created or revised are judged.
 * @param source ({ sourceId?: string | null; sourceVersion?: string | null; stale?: Iterable<string> }): the source a replace_source insert replaces, the version it recorded, and the relations that replacement would retire; rivals among them are deferred.
 * @return (Pending[]): the pending judgments written.
 */
export async function markPending(
  runner: GraphPrimitives,
  ns: string,
  written: readonly Relation[],
  caps: ReadonlyMap<string, ops.RelationCap> | null,
  writtenAt: Date,
  source: { sourceId?: string | null; sourceVersion?: string | null; stale?: Iterable<string> } = {},
): Promise<Pending[]> {
  const fresh = new Map(written.filter((r) => r.created_at.getTime() === writtenAt.getTime()).map((r) => [r.id, r]));
  const rivals = await ops.rivalRelations(runner, ns, written, caps, MAX_RIVALS);
  const stale = new Set(source.stale ?? []);
  const sourceVersion = source.sourceVersion ?? null;
  const pending: Pending[] = [];
  for (const [rid, edges] of rivals) {
    const relation = fresh.get(rid);
    if (relation === undefined) continue;
    const batch = [
      ...new Set(written.filter((r) => r.from_entity === relation.from_entity && r.id !== rid).map((r) => r.id)),
    ].sort(comparePy);
    const deferred =
      sourceVersion === null
        ? []
        : [...new Set(edges.map((e) => e.props.id as string).filter((id) => stale.has(id)))].sort(comparePy);
    const item: Pending = {
      relation,
      batch,
      source_id: source.sourceId ?? null,
      source_version: sourceVersion,
      deferred,
    };
    await runner.updateCurrentEdge(ns, rid, { [PENDING]: encode(item) });
    pending.push(item);
  }
  return pending;
}

/**
 * Read the pending judgments left on relation versions from some entities, one group per write, oldest first.
 *
 * @param runner (GraphPrimitives): the graph runner.
 * @param ns (string): the namespace.
 * @param sources (Iterable<string>): the source entity ids.
 * @param skip (ReadonlySet<string>): relation ids to leave out.
 * @return (Pending[][]): the pending judgments, grouped by the write that stored them.
 */
export async function pendingJudgments(
  runner: GraphPrimitives,
  ns: string,
  sources: Iterable<string>,
  skip: ReadonlySet<string>,
): Promise<Pending[][]> {
  const groups = new Map<string, Map<string, Pending>>();
  for (const source of [...new Set(sources)].sort(comparePy)) {
    const filter = { from_label: L_ENTITY, to_label: L_ENTITY, from_id: source };
    for (const edge of await ops.allEdges(runner, ns, filter)) {
      const props = edge.props;
      if (!props[PENDING] || skip.has(props.id as string)) continue;
      const marker = loads<Record<string, unknown>>(props[PENDING], {});
      const item: Pending = {
        relation: ops.relationOf(edge),
        batch: (marker.batch as string[] | undefined) ?? [],
        source_id: (marker.source_id as string | null | undefined) ?? null,
        source_version: (marker.source_version as string | null | undefined) ?? null,
        deferred: (marker.deferred as string[] | undefined) ?? [],
      };
      const key = props.created_at as string;
      const group = groups.get(key) ?? new Map<string, Pending>();
      groups.set(key, group.set(JSON.stringify([props.id, ops.revisionOf(edge)]), item));
    }
  }
  return [...groups].sort((a, b) => comparePy(a[0], b[0])).map(([, group]) => [...group.values()]);
}

/**
 * Ask the LLM which older relations each pending relation makes untrue, outside any write, writing nothing.
 *
 * @param runner (GraphPrimitives): the graph reader.
 * @param ns (string): the namespace.
 * @param llm (LLMClient): the LLM to ask.
 * @param pending (readonly Pending[]): pending judgments from one write.
 * @param caps (ReadonlyMap<string, ops.RelationCap> | null): the schema's max_count limits by relation type.
 * @param names (ReadonlyMap<string, string>): names of entities as the write stated them.
 * @return (Array<readonly [Relation, EdgeRow]>): each judged relation with a rival edge row the LLM judged it contradicts.
 */
export async function askJudge(
  runner: GraphPrimitives,
  ns: string,
  llm: LLMClient,
  pending: readonly Pending[],
  caps: ReadonlyMap<string, ops.RelationCap> | null,
  names: ReadonlyMap<string, string> = new Map(),
): Promise<Array<readonly [Relation, EdgeRow]>> {
  const byId = new Map(pending.filter((p) => p.relation.is_current).map((p) => [p.relation.id, p.relation]));
  const also = pending.flatMap((p) => p.batch);
  const rivals = await ops.rivalRelations(runner, ns, [...byId.values()], caps, MAX_RIVALS, also);
  const refs = new Set<string>();
  for (const [rid, edges] of rivals) {
    refs.add(byId.get(rid)!.from_entity).add(byId.get(rid)!.to_entity);
    for (const edge of edges) refs.add(edge.to_id);
  }
  const nodes = await ops.nodesById(runner, ns, L_ENTITY, [...refs].sort(comparePy));
  const name = (eid: string): string => {
    const props = nodes.get(eid);
    return names.get(eid) ?? (props === undefined ? eid : decodeEntity(props).name);
  };

  const chosen: Array<readonly [Relation, EdgeRow]> = [];
  const taken = new Set<unknown>();
  for (const [rid, edges] of rivals) {
    const fresh = byId.get(rid)!;
    const open = edges.filter((e) => !taken.has(e.props.id));
    if (!open.length) continue;
    const older = open
      .map((e, i) => `${i + 1}. ${line(e.type, name(e.from_id), name(e.to_id), ops.relationOf(e).properties)}`)
      .join("\n");
    const user =
      `New fact: ${line(fresh.type, name(fresh.from_entity), name(fresh.to_entity), fresh.properties)}\n` +
      `Older facts:\n${older}\n` +
      "Which older facts, by number, are no longer true now that the new fact holds?";
    let verdict: z.infer<typeof Verdict>;
    try {
      verdict = Verdict.parse(await llm.generate({ system: SYSTEM, user, output: Verdict }));
    } catch (error) {
      if (!(error instanceof z.ZodError)) throw error;
      throw new ExtractionError("LLM client returned an invalid contradiction verdict", {
        code: "invalid_llm_output",
        field: "retired",
        cause: error,
      });
    }
    for (const number of new Set(verdict.retired)) {
      if (number >= 1 && number <= open.length) {
        const edge = open[number - 1]!;
        taken.add(edge.props.id);
        chosen.push([fresh, edge]);
      }
    }
  }
  return chosen;
}

/**
 * Tell whether an edge row is the stored version a relation was read as.
 *
 * @param edge (EdgeRow): the edge row.
 * @param rel (Relation): the relation as read.
 * @return (boolean): true when the revision and creation time agree.
 */
function same(edge: EdgeRow, rel: Relation): boolean {
  const version = ops.relationOf(edge);
  return version.revision === rel.revision && version.created_at.getTime() === rel.created_at.getTime();
}

/**
 * Tell whether two edge rows are the same stored version, so a relation deleted and created again does not match.
 *
 * @param a (EdgeRow): one edge row.
 * @param b (EdgeRow): the other edge row.
 * @return (boolean): true when their revision and creation time agree.
 */
function sameVersion(a: EdgeRow, b: EdgeRow): boolean {
  return ops.revisionOf(a) === ops.revisionOf(b) && a.props.created_at === b.props.created_at;
}

/**
 * Read a relation's current edge row under the write's claim, retrying while another write is writing it.
 *
 * @param runner (GraphPrimitives): the graph runner, inside the write.
 * @param ns (string): the namespace.
 * @param rid (string): the relation id.
 * @return (EdgeRow | null): the current edge row, or null when the relation has no current version.
 */
async function claimedCurrent(runner: GraphPrimitives, ns: string, rid: string): Promise<EdgeRow | null> {
  const claim = ops.takeClaim(runner, ns, rid);
  const edge = await runner.getCurrentEdge(ns, rid);
  if (edge === null) {
    ops.requireUnshared(claim, rid, "relation");
    return null;
  }
  ops.foundCurrent(claim);
  return edge;
}

/**
 * Retire what the judge chose and settle the pending judgments, skipping any relation changed since it was judged.
 *
 * @param runner (GraphPrimitives): the graph runner, inside the write.
 * @param ns (string): the namespace.
 * @param pending (readonly Pending[]): the pending judgments from one write.
 * @param chosen (ReadonlyArray<readonly [Relation, EdgeRow]> | null): the verdicts from askJudge, or null when the judge failed, which keeps the judgments pending.
 * @param known (Date): the timestamp of this write, stamped on each retirement.
 * @return ([InsertWarning[], string[]]): the warnings, and the relations the deferred source replacements retired.
 */
export async function applyVerdicts(
  runner: GraphPrimitives,
  ns: string,
  pending: readonly Pending[],
  chosen: ReadonlyArray<readonly [Relation, EdgeRow]> | null,
  known: Date,
): Promise<[InsertWarning[], string[]]> {
  for (const source of ops.dedupList(pending.map((p) => p.relation.from_entity)).sort(comparePy)) {
    await runner.lockNode(ns, L_ENTITY, source);
  }
  const live = new Set<string>();
  const held = new Map<string, readonly [Pending, EdgeRow]>();
  for (const item of pending) {
    const rel = item.relation;
    let edge = await claimedCurrent(runner, ns, rel.id);
    if (edge === null || !same(edge, rel)) {
      edge = (await runner.listEdgeVersions(ns, rel.id)).find((v) => same(v, rel)) ?? null;
    } else if (edge.props[PENDING]) {
      live.add(rel.id);
    }
    if (edge !== null && edge.props[PENDING]) held.set(rel.id, [item, edge]);
  }
  const retirements: Array<readonly [EdgeRow, string, Date]> = [];
  const kept: InsertWarning[] = [];
  for (const [asked, judged] of chosen ?? []) {
    if (!live.has(asked.id)) continue;
    const edge = await claimedCurrent(runner, ns, judged.props.id as string);
    if (edge === null || !sameVersion(edge, judged)) continue;
    if (edge.props.certified_at === null || edge.props.certified_at === undefined) {
      retirements.push([edge, asked.id, ops.effectiveTime(asked)]);
      continue;
    }
    const message = `relation ${edge.props.id} is certified, so it stays current though the LLM judged ${asked.id} to contradict it`;
    kept.push(insertWarning("certified_kept", message));
  }
  const retired = await ops.retireRelations(runner, ns, retirements, known);
  const warnings = [
    ...kept,
    ...retired.map((rel) =>
      insertWarning(
        "relation_superseded",
        `relation ${rel.id} was superseded by ${rel.invalidated_by}, which the LLM judged to contradict it`,
      ),
    ),
  ];
  const deferred = new Map<string, [string, string, Set<string>]>();
  for (const [item] of held.values()) {
    if (!item.deferred.length || item.source_id === null || item.source_version === null) continue;
    const key = JSON.stringify([item.source_id, item.source_version]);
    const entry = deferred.get(key) ?? [item.source_id, item.source_version, new Set<string>()];
    for (const rid of item.deferred) entry[2].add(rid);
    deferred.set(key, entry);
  }
  const bySource: string[] = [];
  for (const [, [sourceId, version, stale]] of [...deferred].sort((a, b) => comparePy(a[0], b[0]))) {
    const [gone, notes] = await ops.replaceSourceLate(runner, ns, sourceId, version, stale, known);
    bySource.push(...gone);
    warnings.push(...notes);
  }
  for (const [rid, [item, edge]] of held) {
    const marker = chosen === null && live.has(rid) ? encode({ ...item, deferred: [] }) : null;
    if (live.has(rid)) await runner.updateCurrentEdge(ns, rid, { [PENDING]: marker });
    else
      await runner.createEdges(ns, L_ENTITY, L_ENTITY, [
        [edge.type, edge.from_id, edge.to_id, { ...edge.props, [PENDING]: marker }],
      ]);
  }
  return [warnings, bySource];
}

/**
 * Judge a committed write's pending relations and any left pending from the same entities, oldest first.
 *
 * @param graph (GraphAccess): the graph, for the read and the short write that follow the committed one.
 * @param llm (LLMClient): the judge.
 * @param written (readonly Relation[]): the relations the write stored.
 * @param pending (readonly Pending[]): the pending judgments the write marked.
 * @param caps (ReadonlyMap<string, ops.RelationCap> | null): the schema's max_count limits by relation type.
 * @param names (ReadonlyMap<string, string>): names of entities as the write stated them.
 * @return ([InsertWarning[], string[]]): the warnings, and the relations deferred source replacements retired.
 */
export async function judgeWritten(
  graph: GraphAccess,
  llm: LLMClient,
  written: readonly Relation[],
  pending: readonly Pending[],
  caps: ReadonlyMap<string, ops.RelationCap> | null,
  names: ReadonlyMap<string, string> = new Map(),
): Promise<[InsertWarning[], string[]]> {
  const mine = new Set(pending.map((p) => p.relation.id));
  const left = await graph.read((store, ns) =>
    pendingJudgments(
      store,
      ns,
      written.map((r) => r.from_entity),
      mine,
    ),
  );
  const warnings: InsertWarning[] = [];
  const retired: string[] = [];
  const groups: Array<readonly [readonly Pending[], ReadonlyMap<string, string>]> = [
    ...left.map((g) => [g, new Map<string, string>()] as const),
    [pending, names] as const,
  ];
  for (const [group, stated] of groups) {
    if (!group.length) continue;
    const [notes, gone] = await settle(graph, llm, group, caps, stated);
    warnings.push(...notes);
    retired.push(...gone);
  }
  return [warnings, retired];
}

/**
 * Ask the judge about one write's pending relations, then apply its verdicts in a short write.
 *
 * @param graph (GraphAccess): the graph.
 * @param llm (LLMClient): the judge.
 * @param group (readonly Pending[]): the pending judgments from one write.
 * @param caps (ReadonlyMap<string, ops.RelationCap> | null): the schema's max_count limits by relation type.
 * @param names (ReadonlyMap<string, string>): names of entities as the write stated them.
 * @return ([InsertWarning[], string[]]): the warnings, and the relations deferred source replacements retired.
 */
async function settle(
  graph: GraphAccess,
  llm: LLMClient,
  group: readonly Pending[],
  caps: ReadonlyMap<string, ops.RelationCap> | null,
  names: ReadonlyMap<string, string>,
): Promise<[InsertWarning[], string[]]> {
  const ids = group
    .map((p) => p.relation.id)
    .sort(comparePy)
    .join(", ");
  const failed: InsertWarning[] = [];
  let chosen: Array<readonly [Relation, EdgeRow]> | null = null;
  try {
    chosen = await graph.read((store, ns) => askJudge(store, ns, llm, group, caps, names));
  } catch (exc) {
    const message = `the LLM contradiction judge failed, so ${ids} stay pending: ${summarizeError(exc)}`;
    failed.push(insertWarning("contradiction_not_judged", message));
  }
  try {
    const [notes, retired] = await graph.write((tx, ns) => applyVerdicts(tx, ns, group, chosen, new Date()));
    return [[...failed, ...notes], retired];
  } catch (exc) {
    if (!(exc instanceof TransientError) && !(exc instanceof StoreError)) throw exc;
    const message = `the LLM contradiction verdicts on ${ids} were not applied, so they stay pending: ${exc.message}`;
    return [[...failed, insertWarning("contradiction_not_judged", message)], []];
  }
}
