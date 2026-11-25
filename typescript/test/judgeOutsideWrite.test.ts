import { AsyncLocalStorage } from "node:async_hooks";
import { expect, it } from "vitest";
import type { z } from "zod";

import { PENDING } from "../src/contradiction.js";
import {
  type GraphPrimitives,
  type InsertResult,
  ids,
  type LLMClient,
  SelfGraph,
  StoreError,
  TransientError,
} from "../src/index.js";
import { LLMResolver, type ResolutionDecision } from "../src/resolution.js";
import { captureLog, forEachStore, freshGraphs, type StoreSpec, worksAt } from "./helpers.js";

const ACME = ids.relationId("person:ann", "WORKS_AT", "company:acme");
const GLOBEX = ids.relationId("person:ann", "WORKS_AT", "company:globex");
const ANN_ACME_GLOBEX = [
  { type: "Person", name: "Ann" },
  { type: "Company", name: "Acme" },
  { type: "Company", name: "Globex" },
];
const KATHY = { entities: [{ type: "Person", name: "Kathy Jones" }] };
const annAt = (company: string, year: number, properties?: Record<string, unknown>) =>
  worksAt("person:ann", `company:${company}`, year, properties);

class Judge implements LLMClient {
  readonly prompts: string[] = [];
  constructor(private readonly retire: number[] = [1]) {}

  async generate<T>(args: { system: string; user: string; output: z.ZodType<T> }): Promise<T> {
    this.prompts.push(args.user);
    return args.output.parse({ retired: this.retire });
  }
}

class FailingJudge implements LLMClient {
  readonly prompts: string[] = [];

  async generate<T>(args: { system: string; user: string; output: z.ZodType<T> }): Promise<T> {
    this.prompts.push(args.user);
    throw new Error("provider down");
  }
}

class WritingLLM implements LLMClient {
  readonly prompts: string[] = [];
  write: (() => Promise<unknown>) | null = null;
  private readonly otherClient = AsyncLocalStorage.snapshot();
  constructor(private readonly answer: Record<string, unknown>) {}

  async generate<T>(args: { system: string; user: string; output: z.ZodType<T> }): Promise<T> {
    this.prompts.push(args.user);
    const write = this.write;
    if (write !== null) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("the write waited on the LLM call")), 5000);
      });
      await Promise.race([this.otherClient(() => write()), timeout]).finally(() => clearTimeout(timer));
    }
    return args.output.parse(this.answer);
  }
}

class PausedJudge implements LLMClient {
  release: () => void = () => {};
  private readonly released = new Promise<void>((resolve) => {
    this.release = resolve;
  });
  private markAsked: () => void = () => {};
  readonly asked = new Promise<void>((resolve) => {
    this.markAsked = resolve;
  });

  async generate<T>(args: { system: string; user: string; output: z.ZodType<T> }): Promise<T> {
    this.markAsked();
    await this.released;
    return args.output.parse({ retired: [] });
  }
}

/**
 * Make the store's transactions after the first `pass` throw an error once each, as many times as asked.
 *
 * @param g (SelfGraph): the graph.
 * @param pass (number): how many transactions run normally first.
 * @param error (() => Error): the error to throw.
 * @param times (number): how many transactions throw.
 * @return (void): nothing.
 */
function failTransactions(g: SelfGraph, pass: number, error: () => Error, times = 1): void {
  const store = g.store as unknown as { transaction: <T>(fn: (tx: GraphPrimitives) => Promise<T>) => Promise<T> };
  const original = store.transaction.bind(store);
  let seen = 0;
  store.transaction = <T>(fn: (tx: GraphPrimitives) => Promise<T>): Promise<T> => {
    seen += 1;
    if (seen > pass && seen <= pass + times) return Promise.reject(error());
    return original(fn);
  };
}

function suite(spec: StoreSpec): void {
  const open = freshGraphs(spec, "judge");

  const annGraph = async (llm: LLMClient): Promise<SelfGraph> => {
    const g = open({ llm, extraction: "regex", contradiction: "llm" });
    await g.insert.facts({ entities: ANN_ACME_GLOBEX, relations: [annAt("acme", 2020)] });
    return g;
  };
  const current = async (g: SelfGraph): Promise<string[]> =>
    (await g.relations.list({ from_entity: "person:ann" })).items.map((r) => r.id).sort();
  const marked = async (g: SelfGraph, rid: string): Promise<boolean> =>
    Boolean((await g.store.getCurrentEdge(g.namespace, rid))?.props[PENDING]);
  const sameGraph = (g: SelfGraph, llm: LLMClient): SelfGraph =>
    new SelfGraph(g.store, { namespace: g.namespace, llm, extraction: "regex", contradiction: "llm" });

  it("asks the LLM judge outside the write and rechecks what it chose", async () => {
    const judge = new WritingLLM({ retired: [1] });
    const g = await annGraph(judge);
    judge.write = () => g.relations.certify(ACME, { by: "hr" });
    const res = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect([judge.prompts.length, res.warnings.map((w) => w.code)]).toEqual([1, ["certified_kept"]]);
    expect(await current(g)).toEqual([ACME, GLOBEX].sort());
    expect(await marked(g, GLOBEX)).toBe(false);
  });

  it("drops a judge verdict when the new relation changed while the LLM answered", async () => {
    const judge = new WritingLLM({ retired: [1] });
    const g = await annGraph(judge);
    judge.write = () => g.relations.update(GLOBEX, { set_properties: { role: "intern" } });
    const res = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect([judge.prompts.length, res.warnings]).toEqual([1, []]);
    expect(await current(g)).toEqual([ACME, GLOBEX].sort());
    expect((await g.relations.get(GLOBEX))!.revision).toBe(2);
  });

  it.each(["rival", "new"])("drops a judge verdict when the %s relation is created again meanwhile", async (which) => {
    const judge = new WritingLLM({ retired: [1] });
    const g = await annGraph(judge);
    const [company, rid] = which === "rival" ? ["acme", ACME] : ["globex", GLOBEX];
    judge.write = async () => {
      await g.relations.delete(rid);
      await g.relations.create("person:ann", "WORKS_AT", `company:${company}`, {
        properties: { role: "temp" },
        contradiction: "warn",
      });
    };
    const res = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect([judge.prompts.length, res.warnings]).toEqual([1, []]);
    const now = new Map((await g.relations.list({ from_entity: "person:ann" })).items.map((r) => [r.id, r]));
    expect([...now.keys()].sort()).toEqual([ACME, GLOBEX].sort());
    expect([now.get(rid)!.revision, now.get(rid)!.properties]).toEqual([1, { role: "temp" }]);
  });

  it("shows a judged write before its verdict and knows the retirement only after", async () => {
    const g = await annGraph(new Judge());
    const res = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect(res.warnings.map((w) => w.code)).toEqual(["relation_superseded"]);
    const old = (await g.relations.versions(ACME)).at(-1)!;
    expect([old.is_current, old.invalidated_by]).toEqual([false, GLOBEX]);
    expect(old.invalidated_known_at!.getTime()).toBeGreaterThan(res.created_at.getTime());
    const window = (await g.retrieve("Ann", { known_at: res.created_at })).subgraph.relations;
    expect(window.map((r) => r.id).sort()).toEqual([ACME, GLOBEX].sort());
    expect((await g.retrieve("Ann")).subgraph.relations.map((r) => r.id)).toEqual([GLOBEX]);
    expect(await marked(g, GLOBEX)).toBe(false);
  });

  it.each(["insert", "create"])("warns on a failed judge and leaves the judgment pending (%s)", async (via) => {
    const g = await annGraph(new FailingJudge());
    const { result, logged } = await captureLog(() =>
      via === "insert"
        ? g.insert.facts({ relations: [annAt("globex", 2022)] }).then((r) => r.warnings)
        : g.relations.create("person:ann", "WORKS_AT", "company:globex", { event_time: "2022-01-01" }).then(() => []),
    );
    const message = `the LLM contradiction judge failed, so ${GLOBEX} stay pending: Error: provider down`;
    if (via === "insert")
      expect(result.map((w) => [w.code, w.message])).toEqual([["contradiction_not_judged", message]]);
    expect(logged.filter((m) => m.includes("contradiction_not_judged"))).toEqual([
      `warning: schema warning [contradiction_not_judged]: ${message}`,
    ]);
    expect(await current(g)).toEqual([ACME, GLOBEX].sort());
    expect(await marked(g, GLOBEX)).toBe(true);
  });

  it.each(["rerun", "other"])("judges a pending relation again on the next judged write (%s)", async (later) => {
    const g = await annGraph(new FailingJudge());
    const first = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect(await marked(g, GLOBEX)).toBe(true);
    const judge = new Judge();
    const again = sameGraph(g, judge);
    let res: InsertResult;
    if (later === "rerun") {
      res = await again.insert.facts({ relations: [annAt("globex", 2022)] });
    } else {
      await again.entities.create("Paris", "City");
      res = await again.insert.facts({
        relations: [{ from_entity: "person:ann", type: "LIVES_IN", to_entity: "city:paris" }],
      });
    }
    expect(judge.prompts.length).toBe(1);
    expect(judge.prompts[0]).toContain("New fact: Ann -[WORKS_AT]-> Globex");
    expect(res.warnings.map((w) => w.code)).toEqual(["relation_superseded"]);
    const now = await current(g);
    expect([now.includes(GLOBEX), now.includes(ACME)]).toEqual([true, false]);
    const old = (await g.relations.versions(ACME)).at(-1)!;
    expect([old.invalidated_by, old.invalidated_at]).toEqual([GLOBEX, (await g.relations.get(GLOBEX))!.event_time]);
    expect(old.invalidated_known_at!.getTime()).toBeGreaterThan(first.created_at.getTime());
    expect(await marked(g, GLOBEX)).toBe(false);
    await again.insert.facts({ relations: [annAt("globex", 2022)] });
    expect(judge.prompts.length).toBe(1);
    await again.close();
  });

  it.each([
    ["TransientError", () => new TransientError("write contention"), 12],
    ["StoreError", () => new StoreError("store gone"), 1],
  ] as const)("leaves the judgment pending when the verdict write fails (%s)", async (_, error, times) => {
    const g = await annGraph(new Judge());
    failTransactions(g, 1, error, times);
    const res = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect(res.warnings.map((w) => w.code)).toEqual(["contradiction_not_judged"]);
    expect(res.warnings[0]!.message).toContain(`the LLM contradiction verdicts on ${GLOBEX} were not applied`);
    expect(await current(g)).toEqual([ACME, GLOBEX].sort());
    expect(await marked(g, GLOBEX)).toBe(true);
    const again = await g.insert.facts({ relations: [annAt("globex", 2022)] });
    expect(again.warnings.map((w) => w.code)).toEqual(["relation_superseded"]);
    expect(await current(g)).toEqual([GLOBEX]);
    expect(await marked(g, GLOBEX)).toBe(false);
  });

  it("keeps a pending judgment's own write out of its rivals", async () => {
    const g = open({ llm: new FailingJudge(), extraction: "regex", contradiction: "llm" });
    await g.insert.facts({ entities: ANN_ACME_GLOBEX, relations: [annAt("acme", 2020)] });
    await g.entities.create("Initech", "Company");
    const initech = ids.relationId("person:ann", "WORKS_AT", "company:initech");
    await g.insert.facts({ relations: [annAt("acme", 2020), annAt("globex", 2022)] });
    await g.insert.facts({ relations: [annAt("initech", 2019), annAt("globex", 2022, { r: 1 })] });
    const judge = new Judge([]);
    const again = sameGraph(g, judge);
    await again.insert.facts({ relations: [annAt("acme", 2020)] });
    expect(judge.prompts.length).toBe(1);
    expect(judge.prompts[0]).toContain("New fact: Ann -[WORKS_AT]-> Globex {r=1}");
    expect(judge.prompts[0]).toContain("Ann -[WORKS_AT]-> Acme");
    expect(judge.prompts[0]).not.toContain("Initech");
    expect([await marked(g, GLOBEX), await marked(g, initech)]).toEqual([false, false]);
    await again.close();
  });

  it.each([true, false])("lets the judge see a rival the replaced source would retire (retire=%s)", async (retire) => {
    const g = open({ llm: new Judge(retire ? [1] : []), extraction: "regex", contradiction: "llm" });
    const old = (await g.insert.text("Ann works at Acme.", { source_id: "hr" })).relations[0]!;
    const res = await g.insert.text("Ann works at Globex.", { source_id: "hr", replace_source: true });
    const fresh = res.relations[0]!;
    const codes = res.warnings.map((w) => w.code).filter((code) => code !== "basic_extraction");
    const stored = (await g.relations.versions(old.id)).find((v) => v.invalidated_at !== null)!;
    if (retire) {
      expect([codes, res.source_diff!.retired, stored.invalidated_by]).toEqual([["relation_superseded"], [], fresh.id]);
    } else {
      expect([codes, res.source_diff!.retired, stored.invalidated_by]).toEqual([
        ["fact_retired_by_source"],
        [old.id],
        res.source_version,
      ]);
    }
    expect((await g.relations.list()).items.map((r) => r.id)).toEqual([fresh.id]);
    expect(await marked(g, fresh.id)).toBe(false);
  });

  it("lets the replaced source retire when the judge fails, keeping the judgment pending", async () => {
    const g = open({ llm: new FailingJudge(), extraction: "regex", contradiction: "llm" });
    const old = (await g.insert.text("Ann works at Acme.", { source_id: "hr" })).relations[0]!;
    const res = await g.insert.text("Ann works at Globex.", { source_id: "hr", replace_source: true });
    const fresh = res.relations[0]!;
    const codes = res.warnings.map((w) => w.code).filter((code) => code !== "basic_extraction");
    expect(codes).toEqual(["contradiction_not_judged", "fact_retired_by_source"]);
    expect(res.source_diff!.retired).toEqual([old.id]);
    expect((await g.relations.list()).items.map((r) => r.id)).toEqual([fresh.id]);
    expect(await marked(g, fresh.id)).toBe(true);
  });

  it("finishes a source replacement cut off after its write on the next write", async () => {
    const g = open({ llm: new Judge([]), extraction: "regex", contradiction: "llm" });
    const old = (await g.insert.text("Ann works at Acme.", { source_id: "hr" })).relations[0]!;
    failTransactions(g, 1, () => new StoreError("process stopped"));
    const res = await g.insert.text("Ann works at Globex.", { source_id: "hr", replace_source: true });
    const fresh = res.relations[0]!;
    expect((await g.relations.list()).items.map((r) => r.id).sort()).toEqual([old.id, fresh.id].sort());
    await g.insert.text("Ann works at Globex.", { source_id: "hr" });
    expect((await g.relations.list()).items.map((r) => r.id)).toEqual([fresh.id]);
    const stored = (await g.relations.versions(old.id)).find((v) => v.invalidated_at !== null)!;
    expect(stored.invalidated_by).toBe(res.source_version);
  });

  it("folds a deferred source replacement into a newer replacement that committed first", async () => {
    const judge = new WritingLLM({ retired: [] });
    const g = open({ llm: judge, extraction: "regex", contradiction: "llm" });
    await g.insert.text("Ann works at Acme.", { source_id: "hr" });
    const other = new SelfGraph(g.store, { namespace: g.namespace, extraction: "regex" });
    const newer: InsertResult[] = [];
    judge.write = async () => {
      newer.push(await other.insert.text("Ann works at Initech.", { source_id: "hr", replace_source: true }));
    };
    const res = await g.insert.text("Ann works at Globex.", { source_id: "hr", replace_source: true });
    expect([judge.prompts.length, newer.length]).toEqual([1, 1]);
    expect((await g.relations.list()).items.map((r) => r.to_entity)).toEqual(["thing:initech"]);
    const stored = new Map(
      (await g.store.matchNodes(g.namespace, "SourceVersion")).map((v) => [v.id as string, v.replaced_by ?? null]),
    );
    expect([stored.get(newer[0]!.source_version!), stored.get(res.source_version!)]).toEqual([
      null,
      newer[0]!.source_version,
    ]);
    const acme = (await g.relations.list({ include_invalidated: true })).items.find(
      (r) => r.to_entity === "thing:acme",
    );
    expect(acme!.invalidated_by).toBe(newer[0]!.source_version);
    const initech = (await g.store.matchNodes(g.namespace, "SourceVersion")).find((v) => v.id === acme!.invalidated_by);
    expect(acme!.invalidated_known_at!.getTime()).toBeGreaterThanOrEqual(
      new Date(initech!.created_at as string).getTime(),
    );
  });

  const appended = [
    ["globex", false],
    ["globex", true],
    ["initech", false],
    ["initech", true],
  ] as const;
  it.each(appended)("folds a deferred replacement past an append: %s first, restated %s", async (first, restated) => {
    const g = open({ extraction: "regex" });
    await g.insert.text("Ann works at Acme.", { source_id: "hr" });
    const judges = { globex: new PausedJudge(), initech: new PausedJudge() };
    const tasks: Record<string, Promise<InsertResult>> = {};
    const graphs: SelfGraph[] = [];
    for (const company of ["globex", "initech"] as const) {
      const graph = new SelfGraph(g.store, {
        namespace: g.namespace,
        extraction: "regex",
        llm: judges[company],
        contradiction: "llm",
      });
      graphs.push(graph);
      const text = `Ann works at ${company[0]!.toUpperCase()}${company.slice(1)}.`;
      tasks[company] = graph.insert.text(text, { source_id: "hr", replace_source: true });
      await judges[company].asked;
    }
    const text = restated ? "Ann works at Acme. Bob works at Paris." : "Bob works at Paris.";
    const added = await g.insert.text(text, { source_id: "hr" });
    for (const company of [first, first === "globex" ? "initech" : "globex"] as const) {
      judges[company].release();
      await tasks[company];
    }
    const initech = await tasks.initech!;
    const now = (await g.relations.list()).items.map((r) => r.to_entity).sort();
    expect(now).toEqual(restated ? ["thing:acme", "thing:initech", "thing:paris"] : ["thing:initech", "thing:paris"]);
    const stored = new Map(
      (await g.store.matchNodes(g.namespace, "SourceVersion")).map((v) => [v.id as string, v.replaced_by ?? null]),
    );
    expect([stored.get(initech.source_version!), stored.get(added.source_version!)]).toEqual([null, null]);
    if (!restated) {
      const acme = (await g.relations.list({ include_invalidated: true })).items.find(
        (r) => r.to_entity === "thing:acme",
      )!;
      expect(acme.invalidated_by).toBe(initech.source_version);
      const row = (await g.store.matchNodes(g.namespace, "SourceVersion")).find((v) => v.id === initech.source_version);
      expect(acme.invalidated_known_at!.getTime()).toBeGreaterThanOrEqual(
        new Date(row!.created_at as string).getTime(),
      );
    }
    for (const graph of graphs) await graph.close();
  });

  const kathyGraph = async (llm: LLMClient, resolver?: "llm"): Promise<SelfGraph> => {
    const g = open({ llm, ...(resolver && { resolver }) });
    await g.entities.create("Katherine Jones", "Person");
    return g;
  };

  it("asks the LLM resolver before the write and never inside it", async () => {
    const llm = new WritingLLM({ same: true });
    const g = await kathyGraph(llm, "llm");
    llm.write = () => g.entities.create("Unrelated", "Thing");
    const res = await g.insert.facts(KATHY);
    expect(res.merges.map((m) => [m.absorbed_id, m.survivor_id])).toEqual([
      ["person:kathy-jones", "person:katherine-jones"],
    ]);
    expect(llm.prompts.length).toBe(1);
    expect(await g.entities.get("thing:unrelated")).not.toBeNull();
  });

  it.each(["properties", "recreate"])("keeps the lexical decision when the target changed (%s)", async (change) => {
    const llm = new WritingLLM({ same: true });
    const g = await kathyGraph(llm, "llm");
    llm.write = async () => {
      if (change === "properties") {
        await g.entities.update("person:katherine-jones", { properties: { team: "ops" } });
      } else {
        await g.entities.delete("person:katherine-jones");
        await g.entities.create("Katherine Jones", "Person");
      }
    };
    const res = await g.insert.facts(KATHY);
    expect([res.merges, llm.prompts.length]).toEqual([[], 1]);
    expect(await g.entities.get("person:kathy-jones")).not.toBeNull();
    expect(await g.entities.get("person:katherine-jones")).not.toBeNull();
  });

  it("runs an LLM resolver subclass as given, with its private state, asking the LLM once", async () => {
    class Counting extends LLMResolver {
      #seen = 0;
      get seen(): number {
        return this.#seen;
      }
      override async resolve(...args: Parameters<LLMResolver["resolve"]>): Promise<ResolutionDecision> {
        this.#seen += 1;
        return super.resolve(...args);
      }
    }
    const llm = new WritingLLM({ same: true });
    const g = await kathyGraph(llm);
    const resolver = new Counting(llm);
    const res = await g.insert.facts(KATHY, { resolver });
    expect(res.merges.map((m) => [m.absorbed_id, m.survivor_id])).toEqual([
      ["person:kathy-jones", "person:katherine-jones"],
    ]);
    expect([llm.prompts.length, resolver.seen]).toEqual([1, 2]);
  });

  it("keeps an LLM resolver subclass's own resolve", async () => {
    class NeverMerge extends LLMResolver {
      override async resolve(): Promise<ResolutionDecision> {
        return { target_id: null, confidence: 0, reason: "none" };
      }
    }
    const llm = new WritingLLM({ same: true });
    const g = await kathyGraph(llm);
    const res = await g.insert.facts(KATHY, { resolver: new NeverMerge(llm) });
    expect(res.merges).toEqual([]);
    expect(await g.entities.get("person:kathy-jones")).not.toBeNull();
  });

  it("does not ask the LLM resolver again when the write is retried", async () => {
    const llm = new WritingLLM({ same: true });
    const g = await kathyGraph(llm, "llm");
    const store = g.store as unknown as { transaction: <T>(fn: (tx: GraphPrimitives) => Promise<T>) => Promise<T> };
    const original = store.transaction.bind(store);
    let failed = false;
    store.transaction = <T>(fn: (tx: GraphPrimitives) => Promise<T>): Promise<T> =>
      original((tx) => {
        const conflicting = new Proxy(tx, {
          get(target, key, receiver) {
            const value = Reflect.get(target, key, receiver);
            if (key !== "upsertNodes" || failed) return typeof value === "function" ? value.bind(target) : value;
            return () => {
              failed = true;
              throw new TransientError("transaction conflict");
            };
          },
        });
        return fn(conflicting);
      });
    const res = await g.insert.facts(KATHY);
    expect([failed, llm.prompts.length, res.merges.length]).toEqual([true, 1, 1]);
  });
}

await forEachStore("judge outside the write", suite);
