import { z } from "zod";
import type { GraphAccess } from "./access.js";
import { ExtractionError, StoreError, TransientError } from "./errors.js";

import { dumps, loads } from "./internal/json.js";
import { contextPayload } from "./internal/payload.js";
import { comparePy, pyStr } from "./internal/py.js";
import { toIso } from "./internal/time.js";
import { L_ENTITY } from "./labels.js";
import { summarizeError } from "./llm/index.js";
import { type InsertWarning, insertWarning, type JsonObject, type Relation } from "./models.js";
import * as ops from "./ops.js";
import type { EdgeRow, EdgeWrite, GraphPrimitives, LLMClient, Props } from "./protocols.js";
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
  /** Marked versions of the relation that a newer revision replaced, by revision and created_at, settled with it. */
  readonly carried?: ReadonlyArray<readonly [number, string]>;
  /** The created_at of the write that marked it, which orders it among the pending judgments. */
  readonly at?: string | null;
}

/**
 * Encode a pending judgment as the marker stored on its relation version.
 *
 * @param item (Pending): the pending judgment.
 * @return (string): the marker JSON.
 */
function encode(item: Pending): string {
  const marker = {
    batch: item.batch,
    source_id: item.source_id,
    source_version: item.source_version,
    deferred: item.deferred,
  };
  return dumps(item.at === undefined || item.at === null ? marker : { ...marker, at: item.at });
}

/**
 * Find each relation a write is about to create or revise that has a rival, so the write marks it pending.
 *
 * The write stores each mark on the relation version itself (see pendingMarks), in the statement that writes the
 * version, so the relation and its mark commit together even on an auto-commit store.
 *
 * @param runner (GraphPrimitives): the graph runner, inside the write.
 * @param ns (string): the namespace.
 * @param written (readonly Relation[]): the relations the write is about to store, as planned.
 * @param caps (ReadonlyMap<string, ops.RelationCap> | null): the schema's max_count limits by relation type.
 * @param writtenAt (Date): the write's timestamp; only relations it created or revised are judged.
 * @param source ({ sourceId?: string | null; sourceVersion?: string | null; stale?: Iterable<string> }): the source a replace_source insert replaces, the version it recorded, and the relations that replacement would retire; rivals among them are deferred.
 * @return (Pending[]): the pending judgments to mark.
 */
export async function planPending(
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
    pending.push({ relation, batch, source_id: source.sourceId ?? null, source_version: sourceVersion, deferred });
  }
  return pending;
}

/**
 * Give the marker property each pending relation's version is written with.
 *
 * @param pending (Iterable<Pending>): the pending judgments from planPending.
 * @return (Map<string, Props>): each relation id mapped to the properties to add to its new version.
 */
export function pendingMarks(pending: Iterable<Pending>): Map<string, Props> {
  return new Map([...pending].map((item) => [item.relation.id, { [PENDING]: encode(item) }]));
}

/**
 * Read the pending judgments left on relation versions from some entities, one group per write, oldest first.
 *
 * A mark on a version that a newer revision of the same relation replaced, with no mark of its own, is carried to
 * that revision: the judgment is asked about the current version and settles the old mark with it. Any other mark
 * on a version that is no longer current only finishes its deferred source step and is cleared.
 *
 * @param runner (GraphPrimitives): the graph runner.
 * @param ns (string): the namespace.
 * @param sources (Iterable<string>): the source entity ids.
 * @param skip (ReadonlySet<string>): relation versions to leave out, as JSON [id, revision].
 * @return (Pending[][]): the pending judgments, grouped by the write that stored them.
 */
export async function pendingJudgments(
  runner: GraphPrimitives,
  ns: string,
  sources: Iterable<string>,
  skip: ReadonlySet<string>,
): Promise<Pending[][]> {
  const marked: Array<readonly [string, Record<string, unknown>, EdgeRow]> = [];
  const live = new Map<string, EdgeRow>();
  for (const source of [...new Set(sources)].sort(comparePy)) {
    const filter = { from_label: L_ENTITY, to_label: L_ENTITY, from_id: source };
    for (const edge of await ops.allEdges(runner, ns, filter)) {
      const props = edge.props;
      if (props.is_current) live.set(props.id as string, edge);
      if (!props[PENDING] || skip.has(JSON.stringify([props.id, ops.revisionOf(edge)]))) continue;
      const marker = loads<Record<string, unknown>>(props[PENDING], {});
      marked.push([(marker.at as string | undefined) || (props.created_at as string), marker, edge]);
    }
  }
  marked.sort(
    (a, b) =>
      comparePy(a[0], b[0]) ||
      comparePy(a[2].props.id as string, b[2].props.id as string) ||
      ops.revisionOf(a[2]) - ops.revisionOf(b[2]),
  );
  const groups = new Map<string, Map<string, Pending>>();
  const carrying = new Set<string>();
  for (const [at, marker, edge] of marked) {
    const rid = edge.props.id as string;
    let subject = edge;
    let carried: Array<readonly [number, string]> = [];
    const newer = live.get(rid);
    if (
      !edge.props.is_current &&
      newer !== undefined &&
      !newer.props[PENDING] &&
      ops.revisionOf(newer) > ops.revisionOf(edge) &&
      !carrying.has(rid)
    ) {
      subject = newer;
      carried = [[ops.revisionOf(edge), edge.props.created_at as string]];
      carrying.add(rid);
    }
    const item: Pending = {
      relation: ops.relationOf(subject),
      batch: (marker.batch as string[] | undefined) ?? [],
      source_id: (marker.source_id as string | null | undefined) ?? null,
      source_version: (marker.source_version as string | null | undefined) ?? null,
      deferred: (marker.deferred as string[] | undefined) ?? [],
      carried,
      at,
    };
    const group = groups.get(at) ?? new Map<string, Pending>();
    groups.set(at, group.set(JSON.stringify([rid, ops.revisionOf(edge)]), item));
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
    const numbers = [...new Set(verdict.retired)];
    if (numbers.some((number) => number < 1 || number > open.length)) {
      throw new ExtractionError(
        `LLM client returned a contradiction verdict outside 1..${open.length}: [${numbers.join(", ")}]`,
        { code: "invalid_llm_output", field: "retired" },
      );
    }
    for (const number of numbers) {
      const edge = open[number - 1]!;
      taken.add(edge.props.id);
      chosen.push([fresh, edge]);
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
 * Retire what the judge chose and settle the pending judgments, keeping pending any the write cannot settle.
 *
 * A judgment stays pending, with its mark on the relation's current version, when the judge failed, when a rival it
 * chose changed since it was judged, or when a newer revision replaced the judged relation; the next judged write from
 * the entity asks again about the versions then current. A mark on a relation with no current version, or one
 * deleted and created again, is cleared after its deferred source step.
 *
 * @param runner (GraphPrimitives): the graph runner, inside the write.
 * @param ns (string): the namespace.
 * @param pending (readonly Pending[]): the pending judgments from one write.
 * @param chosen (ReadonlyArray<readonly [Relation, EdgeRow]> | null): the verdicts from askJudge, or null when the judge failed, which keeps the judgments pending.
 * @param known (Date): the timestamp of this write, stamped on each retirement.
 * @return ([InsertWarning[], string[], boolean]): the warnings, the relations the deferred source replacements retired, and whether every judgment settled.
 */
export async function applyVerdicts(
  runner: GraphPrimitives,
  ns: string,
  pending: readonly Pending[],
  chosen: ReadonlyArray<readonly [Relation, EdgeRow]> | null,
  known: Date,
): Promise<[InsertWarning[], string[], boolean]> {
  for (const source of ops.dedupList(pending.map((p) => p.relation.from_entity)).sort(comparePy)) {
    await runner.lockNode(ns, L_ENTITY, source);
  }
  const live = new Map<string, EdgeRow>();
  const moved = new Map<string, EdgeRow>();
  const held = new Map<string, readonly [Pending, EdgeRow[]]>();
  for (const item of pending) {
    const rel = item.relation;
    const carried = item.carried ?? [];
    const edge = await claimedCurrent(runner, ns, rel.id);
    let marks: EdgeRow[] = [];
    if (edge !== null && same(edge, rel)) {
      if (edge.props[PENDING] || carried.length) live.set(rel.id, edge);
    } else {
      if (edge !== null && ops.revisionOf(edge) > rel.revision && !edge.props[PENDING]) moved.set(rel.id, edge);
      marks = (await runner.listEdgeVersions(ns, rel.id)).filter((v) => same(v, rel));
    }
    if (carried.length) {
      const versions = await runner.listEdgeVersions(ns, rel.id);
      marks.push(
        ...versions.filter((v) => carried.some(([rev, at]) => ops.revisionOf(v) === rev && v.props.created_at === at)),
      );
    }
    marks = marks.filter((v) => v.props[PENDING]);
    if (live.has(rel.id) || marks.length) held.set(rel.id, [item, marks]);
  }
  const retirements: Array<readonly [EdgeRow, string, Date]> = [];
  const kept: InsertWarning[] = [];
  const changed = new Set<string>();
  for (const [asked, judged] of chosen ?? []) {
    if (!live.has(asked.id)) continue;
    const edge = await claimedCurrent(runner, ns, judged.props.id as string);
    if (edge === null) continue;
    if (!sameVersion(edge, judged)) {
      changed.add(asked.id);
      continue;
    }
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
  const stale = [...new Set([...changed, ...[...moved.keys()].filter((rid) => held.has(rid))])].sort(comparePy);
  if (stale.length) {
    const message = `the LLM answered about versions that have since changed, so ${stale.join(", ")} stay pending`;
    warnings.push(insertWarning("contradiction_not_judged", message));
  }
  const deferred = new Map<string, [string, string, Set<string>]>();
  for (const [item] of held.values()) {
    if (!item.deferred.length || item.source_id === null || item.source_version === null) continue;
    const key = JSON.stringify([item.source_id, item.source_version]);
    const entry = deferred.get(key) ?? [item.source_id, item.source_version, new Set<string>()];
    for (const rid of item.deferred) entry[2].add(rid);
    deferred.set(key, entry);
  }
  const bySource: string[] = [];
  for (const [, [sourceId, version, staleIds]] of [...deferred].sort((a, b) => comparePy(a[0], b[0]))) {
    const [gone, notes] = await ops.replaceSourceLate(runner, ns, sourceId, version, staleIds, known);
    bySource.push(...gone);
    warnings.push(...notes);
  }
  let settled = true;
  for (const [rid, [item, marks]] of held) {
    const open = (live.has(rid) && (chosen === null || changed.has(rid))) || moved.has(rid);
    settled &&= !open;
    const at = item.at ?? toIso(item.relation.created_at)!;
    const marker = open ? encode({ ...item, deferred: [], carried: [], at }) : null;
    const target = live.get(rid) ?? moved.get(rid);
    if (target !== undefined && (open || target.props[PENDING]))
      await runner.updateCurrentEdge(ns, rid, { [PENDING]: marker });
    const rows: EdgeWrite[] = marks.map((v) => [v.type, v.from_id, v.to_id, { ...v.props, [PENDING]: null }]);
    if (rows.length) await runner.createEdges(ns, L_ENTITY, L_ENTITY, rows);
  }
  return [warnings, bySource, settled];
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
  const stored = new Map(written.map((r) => [r.id, r]));
  const own = pending.map((p): Pending => {
    const rel = stored.get(p.relation.id);
    return rel !== undefined && rel.revision === p.relation.revision ? { ...p, relation: rel } : p;
  });
  const mine = new Set(own.map((p) => JSON.stringify([p.relation.id, p.relation.revision])));
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
    [own, names] as const,
  ];
  const blocked = new Set<string>();
  const waiting = new Set<string>();
  for (const [group, stated] of groups) {
    if (!group.length) continue;
    const sources = new Set(group.map((p) => p.relation.from_entity));
    const held = [...sources].some((s) => blocked.has(s));
    if (held) for (const p of group) waiting.add(p.relation.id);
    const [notes, gone, settled] = await settle(graph, llm, group, caps, stated, !held);
    warnings.push(...notes);
    retired.push(...gone);
    if (!settled) for (const s of sources) blocked.add(s);
  }
  if (waiting.size) {
    const ids = [...waiting].sort(comparePy).join(", ");
    const message = `an older judgment from the same entity is still pending, so ${ids} stay pending behind it`;
    warnings.push(insertWarning("contradiction_not_judged", message));
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
 * @param ask (boolean): false to leave the judgments pending without asking, still finishing deferred source steps.
 * @return ([InsertWarning[], string[], boolean]): the warnings, the relations deferred source replacements retired, and whether every judgment settled.
 */
async function settle(
  graph: GraphAccess,
  llm: LLMClient,
  group: readonly Pending[],
  caps: ReadonlyMap<string, ops.RelationCap> | null,
  names: ReadonlyMap<string, string>,
  ask = true,
): Promise<[InsertWarning[], string[], boolean]> {
  const ids = group
    .map((p) => p.relation.id)
    .sort(comparePy)
    .join(", ");
  const failed: InsertWarning[] = [];
  let chosen: Array<readonly [Relation, EdgeRow]> | null = null;
  try {
    if (ask) chosen = await graph.read((store, ns) => askJudge(store, ns, llm, group, caps, names));
  } catch (exc) {
    const message = `the LLM contradiction judge failed, so ${ids} stay pending: ${summarizeError(exc)}`;
    failed.push(insertWarning("contradiction_not_judged", message));
  }
  try {
    const [notes, retired, settled] = await graph.write((tx, ns) => applyVerdicts(tx, ns, group, chosen, new Date()));
    return [[...failed, ...notes], retired, settled];
  } catch (exc) {
    if (!(exc instanceof TransientError) && !(exc instanceof StoreError)) throw exc;
    const message = `the LLM contradiction verdicts on ${ids} were not applied, so they stay pending: ${exc.message}`;
    return [[...failed, insertWarning("contradiction_not_judged", message)], [], false];
  }
}
