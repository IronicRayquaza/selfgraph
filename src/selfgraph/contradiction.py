from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, StrictInt
from pydantic import ValidationError as PydanticValidationError

from . import _ops
from ._payload import context_payload
from ._serde import L_ENTITY, _dumps, _loads, decode_entity
from .errors import ExtractionError
from .models import InsertWarning, Relation
from .protocols import LLMClient

__all__ = ["PENDING", "Pending", "apply_verdicts", "ask_judge", "pending_judgments", "pending_marks", "plan_pending"]

_MAX_RIVALS = 20
PENDING = "judge_pending_json"
_SYSTEM = (
    "You decide whether a new fact makes older facts about the same subject no longer true. List an older fact "
    "only when it and the new fact cannot both hold at once, such as a previous employer after a job change or "
    "a liking that became a dislike. Facts that can hold together, such as knowing several people, are not "
    "contradicted. When unsure, list nothing."
)


class _Verdict(BaseModel):
    retired: list[StrictInt] = Field(default_factory=list)


def _line(edge_type: str, source: str, target: str, properties: Mapping[str, Any]) -> str:
    """Render one fact for the judge.

    @param edge_type (str): the relation type.
    @param source (str): the source entity's name.
    @param target (str): the destination entity's name.
    @param properties (Mapping[str, Any]): the relation's properties.
    @return (str): the fact as one line, with line breaks and control characters escaped.
    """
    props = " {" + ", ".join(f"{k}={v}" for k, v in sorted(properties.items())) + "}" if properties else ""
    return context_payload(f"{source} -[{edge_type}]-> {target}{props}")


class Pending(NamedTuple):
    relation: Relation
    batch: list[str]
    source_id: str | None
    source_version: str | None
    deferred: list[str]
    carried: tuple[tuple[int, str], ...] = ()
    at: str | None = None


def _encode(item: Pending) -> str:
    """Encode a pending judgment as the marker stored on its relation version.

    @param item (Pending): the pending judgment.
    @return (str): the marker JSON.
    """
    marker = {
        "batch": item.batch,
        "source_id": item.source_id,
        "source_version": item.source_version,
        "deferred": item.deferred,
    }
    return _dumps({**marker, "at": item.at} if item.at is not None else marker)


async def plan_pending(
    runner: Any,
    ns: str,
    written: list[Relation],
    caps: Mapping[str, _ops.RelationCap] | None,
    written_at: datetime,
    *,
    source_id: str | None = None,
    source_version: str | None = None,
    stale: Iterable[str] = (),
) -> list[Pending]:
    """Find each relation a write is about to create or revise that has a rival, so the write marks it pending.

    The write stores each mark on the relation version itself (see pending_marks), in the statement that writes the
    version, so the relation and its mark commit together even on an auto-commit store.

    @param runner (Any): graph runner, inside the write.
    @param ns (str): namespace.
    @param written (list[Relation]): the relations the write is about to store, as planned.
    @param caps (Mapping[str, RelationCap] | None): the schema's max_count limits by relation type.
    @param written_at (datetime): the write's timestamp; only relations it created or revised are judged.
    @param source_id (str | None): the source a replace_source insert replaces.
    @param source_version (str | None): the version that insert recorded.
    @param stale (Iterable[str]): the relations that replacement would retire; rivals among them are deferred.
    @return (list[Pending]): the pending judgments to mark.
    """
    fresh = {r.id: r for r in written if r.created_at == written_at}
    rivals = await _ops.rival_relations(runner, ns, written, caps, _MAX_RIVALS)
    stale = set(stale)
    pending: list[Pending] = []
    for rid, edges in rivals.items():
        rel = fresh.get(rid)
        if rel is None:
            continue
        batch = sorted({r.id for r in written if r.from_entity == rel.from_entity} - {rid})
        deferred = sorted(stale & {e["props"]["id"] for e in edges}) if source_version is not None else []
        pending.append(Pending(rel, batch, source_id, source_version, deferred))
    return pending


def pending_marks(pending: Iterable[Pending]) -> dict[str, dict[str, str]]:
    """Give the marker property each pending relation's version is written with.

    @param pending (Iterable[Pending]): the pending judgments from plan_pending.
    @return (dict[str, dict[str, str]]): each relation id mapped to the properties to add to its new version.
    """
    return {item.relation.id: {PENDING: _encode(item)} for item in pending}


async def pending_judgments(
    runner: Any, ns: str, sources: Iterable[str], skip: set[tuple[str, int]]
) -> list[list[Pending]]:
    """Read the pending judgments left on relation versions from some entities, one group per write, oldest first.

    A mark on a version that a newer revision of the same relation replaced, with no mark of its own, is carried to
    that revision: the judgment is asked about the current version and settles the old mark with it. Any other mark
    on a version that is no longer current only finishes its deferred source step and is cleared.

    @param runner (Any): graph runner.
    @param ns (str): namespace.
    @param sources (Iterable[str]): the source entity ids.
    @param skip (set[tuple[str, int]]): relation versions to leave out, by id and revision.
    @return (list[list[Pending]]): the pending judgments, grouped by the write that stored them.
    """
    marked: list[tuple[str, dict, dict]] = []
    live: dict[str, dict] = {}
    for source in sorted(set(sources)):
        fetch = partial(runner.match_edges, ns, from_label=L_ENTITY, to_label=L_ENTITY, from_id=source)
        for edge in await _ops._paged(fetch, page=_ops._BATCH):
            props = edge["props"]
            if props.get("is_current"):
                live[props["id"]] = edge
            if props.get(PENDING) and (props["id"], _ops._revision(edge)) not in skip:
                marker = _loads(props[PENDING], {})
                marked.append((marker.get("at") or props["created_at"], marker, edge))
    groups: dict[str, dict[tuple[str, int], Pending]] = {}
    carrying: set[str] = set()
    for at, marker, edge in sorted(marked, key=lambda m: (m[0], m[2]["props"]["id"], _ops._revision(m[2]))):
        rid = edge["props"]["id"]
        subject, carried = edge, ()
        newer = live.get(rid)
        if (
            not edge["props"].get("is_current")
            and newer is not None
            and not newer["props"].get(PENDING)
            and _ops._revision(newer) > _ops._revision(edge)
            and rid not in carrying
        ):
            subject, carried = newer, ((_ops._revision(edge), edge["props"]["created_at"]),)
            carrying.add(rid)
        item = Pending(
            _ops._relation_of(subject),
            list(marker.get("batch") or []),
            marker.get("source_id"),
            marker.get("source_version"),
            list(marker.get("deferred") or []),
            carried,
            at,
        )
        groups.setdefault(at, {})[(rid, _ops._revision(edge))] = item
    return [list(group.values()) for _, group in sorted(groups.items())]


async def ask_judge(
    runner: Any,
    ns: str,
    llm: LLMClient,
    pending: list[Pending],
    caps: Mapping[str, _ops.RelationCap] | None,
    names: Mapping[str, str] | None = None,
) -> list[tuple[Relation, dict]]:
    """Ask the LLM which older relations each pending relation makes untrue, outside any write, writing nothing.

    @param runner (Any): graph reader.
    @param ns (str): namespace.
    @param llm (LLMClient): the LLM to ask.
    @param pending (list[Pending]): pending judgments from one write.
    @param caps (Mapping[str, RelationCap] | None): the schema's max_count limits by relation type.
    @param names (Mapping[str, str] | None): names of entities as the write stated them.
    @return (list[tuple[Relation, dict]]): each judged relation with a rival edge row the LLM judged it contradicts.
    """
    by_id = {p.relation.id: p.relation for p in pending if p.relation.is_current}
    also = {rid for p in pending for rid in p.batch}
    rivals = await _ops.rival_relations(runner, ns, list(by_id.values()), caps, _MAX_RIVALS, also)
    refs = {x for rid in rivals for x in (by_id[rid].from_entity, by_id[rid].to_entity)}
    refs |= {e["to_id"] for edges in rivals.values() for e in edges}
    nodes = await _ops._nodes_by_id(runner, ns, L_ENTITY, sorted(refs))
    names = {**{eid: decode_entity(props).name for eid, props in nodes.items()}, **(names or {})}

    def name(eid: str) -> str:
        return names.get(eid, eid)

    chosen: list[tuple[Relation, dict]] = []
    taken: set[str] = set()
    for rid, edges in rivals.items():
        new = by_id[rid]
        open_edges = [e for e in edges if e["props"]["id"] not in taken]
        if not open_edges:
            continue
        older = "\n".join(
            f"{i}. {_line(e['type'], name(e['from_id']), name(e['to_id']), _ops._relation_of(e).properties)}"
            for i, e in enumerate(open_edges, 1)
        )
        user = (
            f"New fact: {_line(new.type, name(new.from_entity), name(new.to_entity), new.properties)}\n"
            f"Older facts:\n{older}\n"
            "Which older facts, by number, are no longer true now that the new fact holds?"
        )
        try:
            verdict = await llm.generate(system=_SYSTEM, user=user, output=_Verdict)
            verdict = _Verdict.model_validate(verdict)
        except PydanticValidationError as exc:
            raise ExtractionError(
                "LLM client returned an invalid contradiction verdict",
                code="invalid_llm_output",
                field="retired",
            ) from exc
        numbers = list(dict.fromkeys(verdict.retired))
        if any(not 1 <= number <= len(open_edges) for number in numbers):
            raise ExtractionError(
                f"LLM client returned a contradiction verdict outside 1..{len(open_edges)}: {numbers}",
                code="invalid_llm_output",
                field="retired",
            )
        for number in numbers:
            edge = open_edges[number - 1]
            taken.add(edge["props"]["id"])
            chosen.append((new, edge))
    return chosen


def _same(edge: dict, rel: Relation) -> bool:
    """Tell whether an edge row is the stored version a relation was read as.

    @param edge (dict): the edge row.
    @param rel (Relation): the relation as read.
    @return (bool): True when the revision and creation time agree.
    """
    version = _ops._relation_of(edge)
    return (version.revision, version.created_at) == (rel.revision, rel.created_at)


def _version(edge: dict) -> tuple[int, str | None]:
    """Identify the stored version of an edge row, so a relation deleted and created again does not match it.

    @param edge (dict): the edge row.
    @return (tuple[int, str | None]): its revision and creation time.
    """
    return _ops._revision(edge), edge["props"].get("created_at")


async def _claimed_current(runner: Any, ns: str, rid: str) -> dict | None:
    """Read a relation's current edge row under the write's claim, retrying while another write is writing it.

    @param runner (Any): graph runner, inside the write.
    @param ns (str): namespace.
    @param rid (str): the relation id.
    @return (dict | None): the current edge row, or None when the relation has no current version.
    """
    claim = _ops._take_claim(runner, ns, rid)
    edge = await runner.get_current_edge(ns, rid)
    if edge is None:
        _ops._require_unshared(claim, rid, "relation")
        return None
    _ops._found_current(claim)
    return edge


async def apply_verdicts(
    runner: Any,
    ns: str,
    pending: list[Pending],
    chosen: list[tuple[Relation, dict]] | None,
    known: datetime,
) -> tuple[list[InsertWarning], list[str], bool]:
    """Retire what the judge chose and settle the pending judgments, keeping pending any the write cannot settle.

    A judgment stays pending, with its mark on the relation's current version, when the judge failed, when a rival
    it chose changed since it was judged, or when a newer revision replaced the judged relation; the next judged
    write from the entity asks again about the versions then current. A mark on a relation with no current version,
    or one deleted and created again, is cleared after its deferred source step.

    @param runner (Any): graph runner, inside the write.
    @param ns (str): namespace.
    @param pending (list[Pending]): the pending judgments from one write.
    @param chosen (list[tuple[Relation, dict]] | None): the verdicts from ask_judge, or None when the judge failed,
        which keeps the judgments pending.
    @param known (datetime): the timestamp of this write, stamped on each retirement.
    @return (tuple[list[InsertWarning], list[str], bool]): the warnings, the relations the deferred source
        replacements retired, and whether every judgment settled.
    """
    for source in sorted(_ops._dedup_list(p.relation.from_entity for p in pending)):
        await runner.lock_node(ns, L_ENTITY, source)
    live: dict[str, dict] = {}
    moved: dict[str, dict] = {}
    held: dict[str, tuple[Pending, list[dict]]] = {}
    for item in pending:
        rel = item.relation
        edge = await _claimed_current(runner, ns, rel.id)
        marks: list[dict] = []
        if edge is not None and _same(edge, rel):
            if edge["props"].get(PENDING) or item.carried:
                live[rel.id] = edge
        else:
            if edge is not None and _ops._revision(edge) > rel.revision and not edge["props"].get(PENDING):
                moved[rel.id] = edge
            marks = [v for v in await runner.list_edge_versions(ns, rel.id) if _same(v, rel)]
        if item.carried:
            versions = await runner.list_edge_versions(ns, rel.id)
            marks += [v for v in versions if _version(v) in set(item.carried)]
        marks = [v for v in marks if v["props"].get(PENDING)]
        if rel.id in live or marks:
            held[rel.id] = (item, marks)
    retirements: list[tuple[dict, str, datetime]] = []
    kept: list[InsertWarning] = []
    changed: dict[str, None] = {}
    for asked, judged in chosen or []:
        if asked.id not in live:
            continue
        edge = await _claimed_current(runner, ns, judged["props"]["id"])
        if edge is None:
            continue
        if _version(edge) != _version(judged):
            changed[asked.id] = None
            continue
        if edge["props"].get("certified_at") is None:
            retirements.append((edge, asked.id, _ops.effective_time(asked)))
            continue
        message = (
            f"relation {edge['props']['id']} is certified, so it stays current though the LLM judged "
            f"{asked.id} to contradict it"
        )
        kept.append(InsertWarning(code="certified_kept", message=message))
    retired = await _ops.retire_relations(runner, ns, retirements, known)
    warnings = kept + [
        InsertWarning(
            code="relation_superseded",
            message=f"relation {rel.id} was superseded by {rel.invalidated_by}, which the LLM judged to contradict it",
        )
        for rel in retired
    ]
    stale = sorted({*changed, *(rid for rid in moved if rid in held)})
    if stale:
        message = f"the LLM answered about versions that have since changed, so {', '.join(stale)} stay pending"
        warnings.append(InsertWarning(code="contradiction_not_judged", message=message))
    deferred: dict[tuple[str, str], set[str]] = {}
    for item, _ in held.values():
        if item.deferred and item.source_id is not None and item.source_version is not None:
            deferred.setdefault((item.source_id, item.source_version), set()).update(item.deferred)
    by_source: list[str] = []
    for (source_id, version), stale_ids in sorted(deferred.items()):
        gone, notes = await _ops.replace_source_late(runner, ns, source_id, version, stale_ids, known)
        by_source.extend(gone)
        warnings.extend(notes)
    settled = True
    for rid, (item, marks) in held.items():
        open_judgment = (rid in live and (chosen is None or rid in changed)) or rid in moved
        settled = settled and not open_judgment
        at = item.at or _ops.to_iso(item.relation.created_at)
        marker = _encode(item._replace(deferred=[], carried=(), at=at)) if open_judgment else None
        target = live.get(rid) or moved.get(rid)
        if target is not None and (open_judgment or target["props"].get(PENDING)):
            await runner.update_current_edge(ns, rid, {PENDING: marker})
        rows = [(v["type"], v["from_id"], v["to_id"], {**v["props"], PENDING: None}) for v in marks]
        if rows:
            await runner.create_edges(ns, L_ENTITY, L_ENTITY, rows)
    return warnings, by_source, settled
