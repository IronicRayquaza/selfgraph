import asyncio
import contextvars
from datetime import datetime

import pytest
from helpers import ANN_ACME_GLOBEX, fresh_graph, run, works_at

from selfgraph import LLMResolver, ResolutionDecision, SelfGraph, StoreError, TransientError, _ops, client, ids
from selfgraph.contradiction import PENDING

ACME = ids.relation_id("person:ann", "WORKS_AT", "company:acme")
GLOBEX = ids.relation_id("person:ann", "WORKS_AT", "company:globex")


class Judge:
    def __init__(self, retire=(1,)):
        self.retire = list(retire)
        self.prompts = []

    async def generate(self, *, system, user, output):
        self.prompts.append(user)
        return output(retired=self.retire)


class WritingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.write = None

    async def generate(self, *, system, user, output):
        self.prompts.append(user)
        if self.write is not None:
            other_client = contextvars.Context().run(asyncio.ensure_future, self.write())
            await asyncio.wait_for(other_client, timeout=5)
        return output(**self.answer)


class FailingJudge:
    def __init__(self):
        self.prompts = []

    async def generate(self, *, system, user, output):
        self.prompts.append(user)
        raise RuntimeError("provider down")


class PausedJudge:
    def __init__(self):
        self.asked = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, *, system, user, output):
        self.asked.set()
        await asyncio.wait_for(self.release.wait(), timeout=5)
        return output(retired=[])


def ann_graph(llm):
    g = fresh_graph("judge", llm=llm, extraction="regex", contradiction="llm")
    run(g.insert.facts({"entities": ANN_ACME_GLOBEX, "relations": [works_at("acme", 2020)]}))
    return g


def current(g):
    return {r.id for r in run(g.relations.list(from_entity="person:ann")).items}


def marked(g, rid):
    edge = run(g.store.get_current_edge(g.namespace, rid))
    return edge is not None and bool(edge["props"].get(PENDING))


def test_llm_judge_answers_outside_the_write_and_rechecks_what_it_chose():
    judge = WritingLLM({"retired": [1]})
    g = ann_graph(judge)
    judge.write = lambda: g.relations.certify(ACME, by="hr")
    res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert len(judge.prompts) == 1 and [w.code for w in res.warnings] == ["certified_kept"]
    assert current(g) == {ACME, GLOBEX} and not marked(g, GLOBEX)


def test_llm_judge_drops_a_verdict_when_the_new_relation_changed_while_it_answered():
    judge = WritingLLM({"retired": [1]})
    g = ann_graph(judge)
    judge.write = lambda: g.relations.update(GLOBEX, set_properties={"role": "intern"})
    res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert len(judge.prompts) == 1 and res.warnings == []
    assert current(g) == {ACME, GLOBEX}
    assert run(g.relations.get(GLOBEX)).revision == 2


@pytest.mark.parametrize("recreated", ["rival", "new"])
def test_llm_judge_drops_a_verdict_when_a_relation_is_created_again_while_it_answered(recreated):
    judge = WritingLLM({"retired": [1]})
    g = ann_graph(judge)
    company, rid = ("acme", ACME) if recreated == "rival" else ("globex", GLOBEX)

    async def recreate():
        await g.relations.delete(rid)
        await g.relations.create(
            "person:ann", "WORKS_AT", f"company:{company}", properties={"role": "temp"}, contradiction="warn"
        )

    judge.write = recreate
    res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert len(judge.prompts) == 1 and res.warnings == []
    now = {r.id: r for r in run(g.relations.list(from_entity="person:ann")).items}
    assert set(now) == {ACME, GLOBEX} and (now[rid].revision, now[rid].properties) == (1, {"role": "temp"})


def test_a_judged_write_is_visible_before_its_verdict_and_known_retired_only_after():
    judge = Judge()
    g = ann_graph(judge)
    res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert [w.code for w in res.warnings] == ["relation_superseded"]
    old = run(g.relations.versions(ACME))[-1]
    assert (old.is_current, old.invalidated_by) == (False, GLOBEX)
    assert old.invalidated_known_at > res.created_at
    window = run(g.retrieve("Ann", known_at=res.created_at)).subgraph.relations
    assert {r.id for r in window} == {ACME, GLOBEX}
    assert {r.id for r in run(g.retrieve("Ann")).subgraph.relations} == {GLOBEX}
    assert not marked(g, GLOBEX)


@pytest.mark.parametrize("via", ["insert", "create"])
def test_a_failed_llm_judge_warns_and_leaves_the_judgment_pending(via, caplog):
    g = ann_graph(FailingJudge())
    with caplog.at_level("WARNING", logger="selfgraph"):
        if via == "insert":
            warnings = run(g.insert.facts({"relations": [works_at("globex", 2022)]})).warnings
        else:
            run(g.relations.create("person:ann", "WORKS_AT", "company:globex", event_time="2022-01-01"))
    message = f"the LLM contradiction judge failed, so {GLOBEX} stay pending: RuntimeError: provider down"
    if via == "insert":
        assert [(w.code, w.message) for w in warnings] == [("contradiction_not_judged", message)]
    logged = [r.getMessage() for r in caplog.records if "contradiction_not_judged" in r.getMessage()]
    assert logged == [f"schema warning [contradiction_not_judged]: {message}"]
    assert current(g) == {ACME, GLOBEX} and marked(g, GLOBEX)


@pytest.mark.parametrize("later", ["rerun", "other"])
def test_the_next_judged_write_from_the_entity_judges_a_pending_relation_again(later):
    g = ann_graph(FailingJudge())
    first = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert marked(g, GLOBEX)
    judge = Judge()
    again = SelfGraph(store=g.store, namespace=g.namespace, llm=judge, extraction="regex", contradiction="llm")
    if later == "rerun":
        res = run(again.insert.facts({"relations": [works_at("globex", 2022)]}))
    else:
        run(again.entities.create("Paris", "City"))
        lives = {"from_entity": "person:ann", "type": "LIVES_IN", "to_entity": "city:paris"}
        res = run(again.insert.facts({"relations": [lives]}))
    assert len(judge.prompts) == 1 and "New fact: Ann -[WORKS_AT]-> Globex" in judge.prompts[0]
    assert [w.code for w in res.warnings] == ["relation_superseded"]
    assert GLOBEX in current(g) and ACME not in current(g)
    old = run(g.relations.versions(ACME))[-1]
    assert (old.invalidated_by, old.invalidated_at) == (GLOBEX, run(g.relations.get(GLOBEX)).event_time)
    assert old.invalidated_known_at > first.created_at
    assert not marked(g, GLOBEX)
    run(again.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert len(judge.prompts) == 1


@pytest.mark.parametrize("failure", [TransientError("write contention"), StoreError("store gone")])
def test_a_verdict_write_that_fails_leaves_the_judgment_pending(monkeypatch, failure):
    judge = Judge()
    g = ann_graph(judge)
    apply = client.apply_verdicts

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(client, "apply_verdicts", fail)
    res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert [w.code for w in res.warnings] == ["contradiction_not_judged"]
    assert f"the LLM contradiction verdicts on {GLOBEX} were not applied" in res.warnings[0].message
    assert current(g) == {ACME, GLOBEX} and marked(g, GLOBEX)
    monkeypatch.setattr(client, "apply_verdicts", apply)
    again = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert [w.code for w in again.warnings] == ["relation_superseded"]
    assert current(g) == {GLOBEX} and not marked(g, GLOBEX)


def test_a_judgment_cut_off_after_its_write_committed_is_judged_by_the_next_write(monkeypatch):
    judge = Judge()
    g = ann_graph(judge)

    async def stopped(self, written, pending, caps, names=None):
        return [], []

    with monkeypatch.context() as patch:
        patch.setattr(SelfGraph, "_judge", stopped)
        res = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert res.warnings == [] and judge.prompts == []
    assert current(g) == {ACME, GLOBEX} and marked(g, GLOBEX)
    again = run(g.insert.facts({"relations": [works_at("globex", 2022)]}))
    assert len(judge.prompts) == 1 and [w.code for w in again.warnings] == ["relation_superseded"]
    assert current(g) == {GLOBEX}


def test_a_pending_judgment_keeps_its_write_out_of_its_rivals():
    g = fresh_graph("judge", llm=FailingJudge(), extraction="regex", contradiction="llm")
    run(g.insert.facts({"entities": ANN_ACME_GLOBEX, "relations": [works_at("acme", 2020)]}))
    run(g.entities.create("Initech", "Company"))
    initech = ids.relation_id("person:ann", "WORKS_AT", "company:initech")
    run(g.insert.facts({"relations": [works_at("acme", 2020), works_at("globex", 2022)]}))
    run(g.insert.facts({"relations": [works_at("initech", 2019), works_at("globex", 2022, properties={"r": 1})]}))
    judge = Judge(retire=())
    again = SelfGraph(store=g.store, namespace=g.namespace, llm=judge, extraction="regex", contradiction="llm")
    run(again.insert.facts({"relations": [works_at("acme", 2020)]}))
    [prompt] = judge.prompts
    assert "New fact: Ann -[WORKS_AT]-> Globex {r=1}" in prompt
    assert "Ann -[WORKS_AT]-> Acme" in prompt and "Initech" not in prompt
    assert not marked(g, GLOBEX) and not marked(g, initech)


@pytest.mark.parametrize("retire", [True, False])
def test_llm_judge_sees_a_rival_the_replaced_source_would_retire(retire):
    g = fresh_graph("judge", llm=Judge([1] if retire else []), extraction="regex", contradiction="llm")
    old = run(g.insert.text("Ann works at Acme.", source_id="hr")).relations[0]
    res = run(g.insert.text("Ann works at Globex.", source_id="hr", replace_source=True))
    new = res.relations[0]
    codes = [w.code for w in res.warnings if w.code != "basic_extraction"]
    stored = [v for v in run(g.relations.versions(old.id)) if v.invalidated_at is not None][0]
    if retire:
        assert codes == ["relation_superseded"] and res.source_diff.retired == []
        assert stored.invalidated_by == new.id
    else:
        assert codes == ["fact_retired_by_source"] and res.source_diff.retired == [old.id]
        assert stored.invalidated_by == res.source_version
    assert [r.id for r in run(g.relations.list()).items] == [new.id]
    assert not marked(g, new.id)


def test_a_failed_judge_still_lets_the_replaced_source_retire_and_keeps_the_judgment_pending():
    g = fresh_graph("judge", llm=FailingJudge(), extraction="regex", contradiction="llm")
    old = run(g.insert.text("Ann works at Acme.", source_id="hr")).relations[0]
    res = run(g.insert.text("Ann works at Globex.", source_id="hr", replace_source=True))
    new = res.relations[0]
    codes = [w.code for w in res.warnings if w.code != "basic_extraction"]
    assert codes == ["contradiction_not_judged", "fact_retired_by_source"]
    assert res.source_diff.retired == [old.id]
    assert [r.id for r in run(g.relations.list()).items] == [new.id] and marked(g, new.id)


def test_a_source_replacement_cut_off_after_its_write_is_finished_by_the_next_write(monkeypatch):
    judge = Judge(retire=())
    g = fresh_graph("judge", llm=judge, extraction="regex", contradiction="llm")
    old = run(g.insert.text("Ann works at Acme.", source_id="hr")).relations[0]

    async def stopped(self, written, pending, caps, names=None):
        return [], []

    with monkeypatch.context() as patch:
        patch.setattr(SelfGraph, "_judge", stopped)
        res = run(g.insert.text("Ann works at Globex.", source_id="hr", replace_source=True))
    new = res.relations[0]
    assert {r.id for r in run(g.relations.list()).items} == {old.id, new.id}
    run(g.insert.text("Ann works at Globex.", source_id="hr"))
    assert [r.id for r in run(g.relations.list()).items] == [new.id]
    stored = [v for v in run(g.relations.versions(old.id)) if v.invalidated_at is not None][0]
    assert stored.invalidated_by == res.source_version


def test_a_deferred_source_replacement_skips_a_version_a_newer_replacement_superseded():
    judge = WritingLLM({"retired": []})
    g = fresh_graph("judge", llm=judge, extraction="regex", contradiction="llm")
    run(g.insert.text("Ann works at Acme.", source_id="hr"))
    other = SelfGraph(store=g.store, namespace=g.namespace, extraction="regex")
    newer = []

    async def initech():
        newer.append(await other.insert.text("Ann works at Initech.", source_id="hr", replace_source=True))

    judge.write = initech
    res = run(g.insert.text("Ann works at Globex.", source_id="hr", replace_source=True))
    assert len(judge.prompts) == 1 and len(newer) == 1
    assert {r.to_entity for r in run(g.relations.list()).items} == {"thing:initech"}
    stored = {v["id"]: v.get("replaced_by") for v in run(g.store.match_nodes(g.namespace, "SourceVersion"))}
    assert stored[newer[0].source_version] is None
    assert stored[res.source_version] == newer[0].source_version
    [acme] = [r for r in run(g.relations.list(include_invalidated=True)).items if r.to_entity == "thing:acme"]
    assert acme.invalidated_by == newer[0].source_version
    [initech] = [v for v in run(g.store.match_nodes(g.namespace, "SourceVersion")) if v["id"] == acme.invalidated_by]
    assert acme.invalidated_known_at >= datetime.fromisoformat(initech["created_at"])


@pytest.mark.parametrize("restated", [False, True])
@pytest.mark.parametrize("first", ["globex", "initech"])
def test_a_deferred_source_replacement_follows_the_newer_one_past_a_version_appended_while_both_wait(first, restated):
    g = fresh_graph("judge", extraction="regex")
    run(g.insert.text("Ann works at Acme.", source_id="hr"))
    judges = {"globex": PausedJudge(), "initech": PausedJudge()}

    async def scenario():
        tasks = {}
        for company in ("globex", "initech"):
            llm = judges[company]
            graph = SelfGraph(store=g.store, namespace=g.namespace, extraction="regex", llm=llm, contradiction="llm")
            text = f"Ann works at {company.capitalize()}."
            tasks[company] = asyncio.ensure_future(graph.insert.text(text, source_id="hr", replace_source=True))
            await asyncio.wait_for(judges[company].asked.wait(), timeout=5)
        text = "Ann works at Acme. Bob works at Paris." if restated else "Bob works at Paris."
        appended = await g.insert.text(text, source_id="hr")
        for company in (first, ({"globex", "initech"} - {first}).pop()):
            judges[company].release.set()
            await tasks[company]
        return await tasks["initech"], appended

    initech, appended = run(scenario())
    now = {r.to_entity for r in run(g.relations.list()).items}
    assert now == {"thing:initech", "thing:paris"} | ({"thing:acme"} if restated else set())
    stored = {v["id"]: v.get("replaced_by") for v in run(g.store.match_nodes(g.namespace, "SourceVersion"))}
    assert (stored[initech.source_version], stored[appended.source_version]) == (None, None)
    if not restated:
        [acme] = [r for r in run(g.relations.list(include_invalidated=True)).items if r.to_entity == "thing:acme"]
        assert acme.invalidated_by == initech.source_version
        [row] = [v for v in run(g.store.match_nodes(g.namespace, "SourceVersion")) if v["id"] == initech.source_version]
        assert acme.invalidated_known_at >= datetime.fromisoformat(row["created_at"])


def kathy_graph(llm, **options):
    g = fresh_graph("judge", llm=llm, **options)
    run(g.entities.create("Katherine Jones", "Person"))
    return g


KATHY = {"entities": [{"type": "Person", "name": "Kathy Jones"}]}


def test_llm_resolver_answers_before_the_write_and_never_asks_inside_it():
    llm = WritingLLM({"same": True})
    g = kathy_graph(llm, resolver="llm")
    llm.write = lambda: g.entities.create("Unrelated", "Thing")
    res = run(g.insert.facts(KATHY))
    assert [(m.absorbed_id, m.survivor_id) for m in res.merges] == [("person:kathy-jones", "person:katherine-jones")]
    assert len(llm.prompts) == 1 and run(g.entities.get("thing:unrelated")) is not None


@pytest.mark.parametrize("change", ["properties", "recreate"])
def test_llm_resolver_keeps_the_lexical_decision_when_the_target_changed_after_asking(change):
    llm = WritingLLM({"same": True})
    g = kathy_graph(llm, resolver="llm")

    async def changed():
        if change == "properties":
            await g.entities.update("person:katherine-jones", properties={"team": "ops"})
        else:
            await g.entities.delete("person:katherine-jones")
            await g.entities.create("Katherine Jones", "Person")

    llm.write = changed
    res = run(g.insert.facts(KATHY))
    assert res.merges == [] and len(llm.prompts) == 1
    assert run(g.entities.get("person:kathy-jones")) is not None
    assert run(g.entities.get("person:katherine-jones")) is not None


def test_an_llm_resolver_subclass_runs_as_given_and_asks_the_llm_once():
    class Counting(LLMResolver):
        calls = 0

        async def resolve(self, candidate, existing):
            self.calls += 1
            return await super().resolve(candidate, existing)

    llm = WritingLLM({"same": True})
    g = kathy_graph(llm)
    resolver = Counting(llm)
    res = run(g.insert.facts(KATHY, resolver=resolver))
    assert [(m.absorbed_id, m.survivor_id) for m in res.merges] == [("person:kathy-jones", "person:katherine-jones")]
    assert (len(llm.prompts), resolver.calls) == (1, 2)


def test_an_llm_resolver_subclass_keeps_its_own_resolve():
    class NeverMerge(LLMResolver):
        async def resolve(self, candidate, existing):
            return ResolutionDecision()

    llm = WritingLLM({"same": True})
    g = kathy_graph(llm)
    res = run(g.insert.facts(KATHY, resolver=NeverMerge(llm)))
    assert res.merges == [] and run(g.entities.get("person:kathy-jones")) is not None


def test_a_retried_write_does_not_ask_the_llm_resolver_again(monkeypatch):
    llm = WritingLLM({"same": True})
    g = kathy_graph(llm, resolver="llm")
    upsert, failed = _ops.upsert_entities, []

    async def conflict_once(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise TransientError("transaction conflict")
        return await upsert(*args, **kwargs)

    monkeypatch.setattr(_ops, "upsert_entities", conflict_once)
    res = run(g.insert.facts(KATHY))
    assert failed and len(llm.prompts) == 1 and len(res.merges) == 1
