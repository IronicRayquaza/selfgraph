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

__all__ = ["PENDING", "Pending", "apply_verdicts", "ask_judge", "mark_pending", "pending_judgments"]

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


def _encode(item: Pending) -> str:
    """Encode a pending judgment as the marker stored on its relation version.

    @param item (Pending): the pending judgment.
    @return (str): the marker JSON.
    """
    return _dumps(
        {
            "batch": item.batch,
            "source_id": item.source_id,
            "source_version": item.source_version,
            "deferred": item.deferred,
        }
    )


async def mark_pending(
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
    """Mark each relation a write created or revised that has a rival as pending judgment, inside that write.

    @param runner (Any): graph runner, inside the write.
    @param ns (str): namespace.
    @param written (list[Relation]): the relations the write stored.
    @param caps (Mapping[str, RelationCap] | None): the schema's max_count limits by relation type.
    @param written_at (datetime): the write's timestamp; only relations it created or revised are judged.
    @param source_id (str | None): the source a replace_source insert replaces.
    @param source_version (str | None): the version that insert recorded.
    @param stale (Iterable[str]): the relations that replacement would retire; rivals among them are deferred.
    @return (list[Pending]): the pending judgments written.
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
        item = Pending(rel, batch, source_id, source_version, deferred)
        await runner.update_current_edge(ns, rid, {PENDING: _encode(item)})
        pending.append(item)
    return pending


async def pending_judgments(runner: Any, ns: str, sources: Iterable[str], skip: set[str]) -> list[list[Pending]]:
    """Read the pending judgments left on relation versions from some entities, one group per write, oldest first.

    @param runner (Any): graph runner.
    @param ns (str): namespace.
    @param sources (Iterable[str]): the source entity ids.
    @param skip (set[str]): relation ids to leave out.
    @return (list[list[Pending]]): the pending judgments, grouped by the write that stored them.
    """
    groups: dict[str, dict[tuple[str, int], Pending]] = {}
    for source in sorted(set(sources)):
        fetch = partial(runner.match_edges, ns, from_label=L_ENTITY, to_label=L_ENTITY, from_id=source)
        for edge in await _ops._paged(fetch, page=_ops._BATCH):
            props = edge["props"]
            if not props.get(PENDING) or props["id"] in skip:
                continue
            marker = _loads(props[PENDING], {})
            item = Pending(
                _ops._relation_of(edge),
                list(marker.get("batch") or []),
                marker.get("source_id"),
                marker.get("source_version"),
                list(marker.get("deferred") or []),
            )
            groups.setdefault(props["created_at"], {})[(props["id"], _ops._revision(edge))] = item
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
        for number in dict.fromkeys(verdict.retired):
            if 1 <= number <= len(open_edges):
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
) -> tuple[list[InsertWarning], list[str]]:
    """Retire what the judge chose and settle the pending judgments, skipping any relation changed since it was judged.

    @param runner (Any): graph runner, inside the write.
    @param ns (str): namespace.
    @param pending (list[Pending]): the pending judgments from one write.
    @param chosen (list[tuple[Relation, dict]] | None): the verdicts from ask_judge, or None when the judge failed,
        which keeps the judgments pending.
    @param known (datetime): the timestamp of this write, stamped on each retirement.
    @return (tuple[list[InsertWarning], list[str]]): the warnings, and the relations the deferred source replacements
        retired.
    """
    for source in sorted(_ops._dedup_list(p.relation.from_entity for p in pending)):
        await runner.lock_node(ns, L_ENTITY, source)
    live: set[str] = set()
    held: dict[str, tuple[Pending, dict]] = {}
    for item in pending:
        rel = item.relation
        edge = await _claimed_current(runner, ns, rel.id)
        if edge is None or not _same(edge, rel):
            versions = await runner.list_edge_versions(ns, rel.id)
            edge = next((v for v in versions if _same(v, rel)), None)
        elif edge["props"].get(PENDING):
            live.add(rel.id)
        if edge is not None and edge["props"].get(PENDING):
            held[rel.id] = (item, edge)
    retirements: list[tuple[dict, str, datetime]] = []
    kept: list[InsertWarning] = []
    for asked, judged in chosen or []:
        if asked.id not in live:
            continue
        edge = await _claimed_current(runner, ns, judged["props"]["id"])
        if edge is None or _version(edge) != _version(judged):
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
    deferred: dict[tuple[str, str], set[str]] = {}
    for item, _ in held.values():
        if item.deferred and item.source_id is not None and item.source_version is not None:
            deferred.setdefault((item.source_id, item.source_version), set()).update(item.deferred)
    by_source: list[str] = []
    for (source_id, version), stale in sorted(deferred.items()):
        gone, notes = await _ops.replace_source_late(runner, ns, source_id, version, stale, known)
        by_source.extend(gone)
        warnings.extend(notes)
    for rid, (item, edge) in held.items():
        marker = _encode(item._replace(deferred=[])) if chosen is None and rid in live else None
        if rid in live:
            await runner.update_current_edge(ns, rid, {PENDING: marker})
        else:
            await runner.create_edges(
                ns,
                L_ENTITY,
                L_ENTITY,
                [(edge["type"], edge["from_id"], edge["to_id"], {**edge["props"], PENDING: marker})],
            )
    return warnings, by_source
