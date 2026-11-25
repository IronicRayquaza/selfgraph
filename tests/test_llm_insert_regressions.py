import pytest
from helpers import FakeLLM, fresh_graph, run

from selfgraph import ids
from selfgraph.contradiction import PENDING


@pytest.fixture
def graph():
    g = fresh_graph(llm=FakeLLM(), extraction="llm", contradiction="llm")
    yield g
    run(g.close())


def extraction(g, properties=None):
    g._llm.responses.update(
        {
            "_EntitiesOut": {"entities": [{"type": "T", "name": n} for n in ("A", "B", "C")]},
            "_RelationsOut": {
                "relations": [{"from_name": "A", "type": "KNOWS", "to_name": "C", "properties": properties or {}}]
            },
        }
    )


@pytest.mark.parametrize("key", ["", "   "])
def test_339_empty_extracted_property(graph, key):
    extraction(graph, {key: "bad", "since": 2020})
    result = run(graph.insert.text("A knows C."))
    assert len(result.entities) == 3
    assert [r.properties for r in result.relations] == [{"since": 2020}]
    [warning] = result.warnings
    assert warning.code == "extraction_partial_failure"
    assert warning.message == "dropped an extracted relation property: properties keys cannot be empty or whitespace"


@pytest.mark.parametrize("operation", ["facts", "text", "create"])
@pytest.mark.parametrize("reply", [{"retired": "bad"}, {"retired": [None]}])
def test_336_invalid_verdict_keeps_the_committed_write_and_its_judgment_pending(graph, operation, reply, caplog):
    run(
        graph.insert.facts(
            {
                "entities": [{"type": "T", "name": n} for n in ("A", "B", "C")],
                "relations": [{"from_entity": "t:a", "type": "KNOWS", "to_entity": "t:b"}],
            }
        )
    )
    graph._llm.responses["_Verdict"] = reply
    extraction(graph)
    with caplog.at_level("WARNING", logger="selfgraph"):
        if operation == "text":
            run(graph.insert.text("A knows C.", source_id="new-source"))
        elif operation == "create":
            run(graph.relations.create("t:a", "KNOWS", "t:c"))
        else:
            run(
                graph.insert.facts(
                    {
                        "entities": [{"type": "T", "name": "D"}],
                        "relations": [{"from_entity": "t:a", "type": "KNOWS", "to_entity": "t:c"}],
                    },
                    source_id="new-source",
                )
            )
    rid = ids.relation_id("t:a", "KNOWS", "t:c")
    assert {r.id for r in run(graph.relations.list()).items} == {ids.relation_id("t:a", "KNOWS", "t:b"), rid}
    edge = run(graph.store.get_current_edge(graph.namespace, rid))
    assert edge["props"].get(PENDING)
    message = (
        f"the LLM contradiction judge failed, so {rid} stay pending: "
        "ExtractionError: LLM client returned an invalid contradiction verdict"
    )
    logged = [r.getMessage() for r in caplog.records if "contradiction_not_judged" in r.getMessage()]
    assert logged == [f"schema warning [contradiction_not_judged]: {message}"]


BLANK_NAME = "dropped an extracted entity: entity name cannot be empty"
BLANK_TYPE = "dropped an extracted entity: entity type cannot be empty"
CONTROL_NAME = "dropped an extracted entity: entity name cannot contain control or surrogate characters"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"type": "Person", "name": ""}, BLANK_NAME),
        ({"type": "Person", "name": "   "}, BLANK_NAME),
        ({"type": "Company", "name": "\t"}, BLANK_NAME),
    ],
)
def test_333_invalid_entity_keeps_the_rest_of_the_chunk_typed(graph, bad, message):
    graph._llm.responses.update(
        {
            "_EntitiesOut": {"entities": [{"type": "Person", "name": "Ann"}, {"type": "Company", "name": "Acme"}, bad]},
            "_RelationsOut": {"relations": [{"from_name": "Ann", "type": "WORKS_AT", "to_name": "Acme"}]},
        }
    )
    result = run(graph.insert.text("Ann works at Acme."))
    stored = {e.id: e.type for e in run(graph.entities.list()).items}
    assert stored == {"person:ann": "Person", "company:acme": "Company"}
    assert [(r.from_entity, r.type, r.to_entity) for r in result.relations] == [
        ("person:ann", "WORKS_AT", "company:acme")
    ]
    assert [(w.code, w.message) for w in result.warnings] == [("extraction_partial_failure", message)]


@pytest.mark.parametrize(
    ("bad", "stored", "messages"),
    [
        ({"type": " ", "name": "Bob"}, {"thing:bob": "Thing"}, [BLANK_NAME, BLANK_TYPE]),
        ({"type": "Person", "name": "Bob\x07"}, {}, [BLANK_NAME, CONTROL_NAME, CONTROL_NAME]),
    ],
)
def test_333_relation_naming_a_dropped_entity_follows_the_unknown_name_rule(graph, bad, stored, messages):
    graph._llm.responses.update(
        {
            "_EntitiesOut": {"entities": [{"type": "Person", "name": "Ann"}, {"type": "Person", "name": ""}, bad]},
            "_RelationsOut": {"relations": [{"from_name": "Ann", "type": "KNOWS", "to_name": bad["name"]}]},
        }
    )
    result = run(graph.insert.text("Ann knows Bob."))
    types = {e.id: e.type for e in run(graph.entities.list()).items}
    assert types == {"person:ann": "Person", **stored}
    assert [r.to_entity for r in result.relations] == list(stored)
    assert [w.message for w in result.warnings] == messages


def test_333_chunk_with_only_invalid_entities_still_inserts(graph):
    graph._llm.responses.update(
        {
            "_EntitiesOut": {
                "entities": [
                    {"type": "Person", "name": ""},
                    {"type": " ", "name": "Ann"},
                    {"type": "Company", "name": "  "},
                    {"type": "", "name": "Cy"},
                ]
            },
            "_RelationsOut": {"relations": [{"from_name": "Ann", "type": "WORKS_AT", "to_name": "Acme"}]},
        }
    )
    result = run(graph.insert.text("Ann works at Acme."))
    assert {e.id: e.type for e in run(graph.entities.list()).items} == {"thing:ann": "Thing", "thing:acme": "Thing"}
    assert [w.message for w in result.warnings] == [BLANK_NAME, BLANK_TYPE, BLANK_NAME, BLANK_TYPE]


def test_484_extracted_plus_and_sharp_names_keep_their_own_entities(graph):
    graph._llm.responses.update(
        {
            "_EntitiesOut": {
                "entities": [{"type": "Language", "name": n} for n in ("C++", "C#", "C")]
                + [{"type": "Person", "name": n} for n in ("Ann", "Bob")]
            },
            "_RelationsOut": {
                "relations": [
                    {"from_name": "Ann", "type": "USES", "to_name": "C++"},
                    {"from_name": "Bob", "type": "USES", "to_name": "C#"},
                ]
            },
        }
    )
    result = run(graph.insert.text("Ann uses C++ and Bob uses C#."))
    langs = {e.id: (e.name, e.aliases) for e in run(graph.entities.list(type="Language")).items}
    assert langs == {"language:c-plus-plus": ("C++", []), "language:c-sharp": ("C#", []), "language:c": ("C", [])}
    assert {r.from_entity: r.to_entity for r in result.relations} == {
        "person:ann": "language:c-plus-plus",
        "person:bob": "language:c-sharp",
    }
