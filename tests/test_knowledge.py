from __future__ import annotations

import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from industrial_diagnosis.fixtures import (
    DECISION_LABELS,
    TREE_PATH_EXAMPLES,
    original_knowledge,
    original_reports,
)
from industrial_diagnosis.knowledge import (
    KnowledgeStore,
    draft_from_reports,
    draft_from_yaml,
    export_owl,
    request_from_reports,
)
from industrial_diagnosis.schemas import DomainError, KnowledgeDraft, PublishRequest
from industrial_diagnosis.storage import Store


@pytest.fixture
def identities(tmp_path):
    store = Store(tmp_path / "knowledge-state")
    tokens = {role: store.provision_user(role, role, ["PLANT-A"]) for role in ("admin", "analyst", "operator")}
    actors = {role: store.authenticate(token) for role, token in tokens.items()}
    outsider = store.authenticate(store.provision_user("other-admin", "admin", ["PLANT-B"]))
    return store, actors, outsider


def configured_java():
    path = os.environ.get("INDUSTRIAL_DIAGNOSIS_JAVA") or os.environ.get("JAVA_PATH")
    if path is None and os.environ.get("JAVA_HOME"):
        path = str(Path(os.environ["JAVA_HOME"]) / "bin" / "java")
    if not path:
        pytest.skip("Real HermiT integration requires an explicitly configured local JVM")
    return path


def publication(value, action_id="publish-original"):
    return PublishRequest(action_id=action_id, version_id=value["version_id"],
                          content_hash=value["content_hash"],
                          review_note="Human review of original synthetic evidence requirements completed.")


def test_original_reports_are_the_actual_draft_source():
    request = original_knowledge()
    assert len(request.reports) == 3
    assert request.draft == draft_from_reports(original_reports())
    assert request == request_from_reports(request.request_id, request.reports)
    assert {r.report_id for r in request.reports} == set(request.draft.source_report_ids)
    assert all(r.synthetic is True for r in request.reports)
    assert request.draft.plant_id == "PLANT-A"
    assert original_knowledge("PLANT-B").draft.plant_id == "PLANT-B"
    assert draft_from_yaml(yaml.safe_dump(request.draft.model_dump()), request.reports) == request.draft


def test_fixture_decisions_have_real_numeric_and_human_requirements():
    draft = original_knowledge().draft
    nodes = {n.node_id: n for n in draft.nodes}
    conditions = {c.condition_id: c for c in draft.conditions}
    assert draft.global_required == ["drive_enabled"]
    assert conditions["drive_enabled"].signal == "drive_enabled"
    assert conditions["drive_enabled"].threshold == 1.0
    assert conditions["drive_enabled"].statistic == "min"
    assert conditions["drive_enabled"].unit == "bool"
    assert nodes["jam"].conditions == ["current_high", "belt_low"]
    assert nodes["jam"].mandatory == ["obstruction_confirmed"]
    assert conditions["current_high"].signal == "motor_current"
    assert conditions["belt_low"].signal == "belt_speed"
    assert conditions["obstruction_confirmed"].source == "human"
    assert all(c.min_samples == 20 and c.max_gap_ms == 1500 for c in draft.conditions)
    for node_id, path in TREE_PATH_EXAMPLES.items():
        assert path[0] == draft.root_id and path[-1] == node_id
        assert nodes[node_id].label == DECISION_LABELS[node_id]
        for parent, child in zip(path, path[1:], strict=False):
            assert child in nodes[parent].children


def test_local_model_extraction_only_returns_an_untrusted_draft():
    reports = original_reports()
    value = original_knowledge().draft.model_dump()

    class LocalModel:
        local_only = True

        def extract(self, sources):
            assert sources == reports
            return yaml.safe_dump(value)

    assert draft_from_reports(reports, extractor=LocalModel()).model_dump() == value
    LocalModel.local_only = False
    with pytest.raises(DomainError, match="explicitly local") as error:
        draft_from_reports(reports, extractor=LocalModel())
    assert error.value.code == "nonlocal_extractor"


def test_local_model_cannot_invent_provenance_or_publish_fields():
    class LocalModel:
        local_only = True

        def extract(self, reports):
            value = original_knowledge().draft.model_dump()
            value["source_report_ids"] = ["fabricated-report"]
            value["publish"] = True
            return value

    with pytest.raises(DomainError) as error:
        draft_from_reports(original_reports(), extractor=LocalModel())
    assert error.value.code == "invalid_draft"


@pytest.mark.parametrize("mutate", [
    lambda v: v["nodes"][0]["children"].append("root"),
    lambda v: v["nodes"][1]["children"].append("normal"),
    lambda v: v["nodes"][0].update(gate="causes"),
    lambda v: v["nodes"][-1].update(conditions=["unknown"]),
    lambda v: v["nodes"][-1].update(children=["normal"]),
    lambda v: v["global_required"].append("drive_enabled"),
    lambda v: v["conditions"][0].update(threshold=float("nan")),
])
def test_pydantic_rejects_invalid_structure_and_conditions(mutate):
    value = original_knowledge().draft.model_dump()
    mutate(value)
    with pytest.raises(ValidationError):
        KnowledgeDraft.model_validate(value)


def test_owl_export_is_local_structured_and_not_a_reasoner_claim():
    text = export_owl(original_knowledge().draft)
    root = ET.fromstring(text)
    owl = "{http://www.w3.org/2002/07/owl#}"
    rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    assert root.find(owl + "Ontology").attrib[rdf + "about"] == "urn:original:conveyor"
    assert root.find(".//" + owl + "imports") is None
    assert root.find(".//" + owl + "AllDisjointClasses") is not None
    assert root.find(".//" + owl + "AllDifferent") is not None
    for token in ("HumanGuardedDiagnosis", "EvidenceBoundLeaf", "hasCondition", "AllGate", "AnyGate",
                  "minimumSamples", "maximumGapMs", "numericThreshold", "condition_obstruction_confirmed"):
        assert token in text


def test_missing_jvm_blocks_publication_but_keeps_provenance(identities):
    store, actors, _ = identities
    knowledge = KnowledgeStore(store, java_path="/nonexistent/local/jvm/bin/java")
    value = knowledge.build(original_knowledge(), actors["analyst"])
    assert value["status"] == value["build_status"] == "blocked"
    assert value["reasoner"]["status"] == "missing_jvm"
    assert value["reasoner"]["consistent"] is None
    assert value["reasoner"]["success"] is False
    assert len(value["provenance"]) == 3
    assert all(len(r["source_sha256"]) == 64 for r in value["provenance"])
    assert value["artifacts"]["classified_owl"] is None
    assert knowledge.get(value["version_id"], actors["operator"]) == value
    with pytest.raises(DomainError) as error:
        knowledge.publish(publication(value), actors["admin"])
    assert error.value.code == "knowledge_not_checked"
    with pytest.raises(DomainError) as error:
        knowledge.active("PLANT-A", actors["admin"])
    assert error.value.code == "no_active_knowledge"


def test_build_idempotency_scope_and_role(identities):
    store, actors, outsider = identities
    knowledge = KnowledgeStore(store, java_path="/nonexistent/java")
    request = original_knowledge()
    with pytest.raises(DomainError) as error:
        knowledge.build(request, actors["operator"])
    assert error.value.code == "forbidden"
    value = knowledge.build(request, actors["analyst"])
    assert knowledge.build(request, actors["analyst"]) == value
    changed = request.model_copy(deep=True)
    changed.draft.title = "Different original title"
    with pytest.raises(DomainError) as error:
        knowledge.build(changed, actors["analyst"])
    assert error.value.code == "idempotency_conflict"
    with pytest.raises(DomainError) as error:
        knowledge.get(value["version_id"], outsider)
    assert error.value.code == "forbidden"
    assert knowledge.list_versions(outsider) == []
    assert len(knowledge.list_versions(actors["admin"])) == 1
    with pytest.raises(DomainError) as error:
        knowledge.publish(publication(value), actors["analyst"])
    assert error.value.code == "forbidden"
    store.set_user("analyst", actors["admin"], enabled=True, plants=["PLANT-A"], role="operator")
    with pytest.raises(DomainError) as error:
        knowledge.get(value["version_id"], actors["analyst"])
    assert error.value.code == "identity_changed"


@pytest.mark.parametrize("artifact", ["draft", "reports", "asserted_owl", "reasoner"])
def test_get_rechecks_every_stored_artifact(identities, artifact):
    store, actors, _ = identities
    knowledge = KnowledgeStore(store, java_path="/nonexistent/java")
    value = knowledge.build(original_knowledge(), actors["admin"])
    reference = value["artifacts"][artifact]
    path = store.artifacts / (reference["sha256"] + ".json")
    path.write_text("{}")
    with pytest.raises(DomainError) as error:
        knowledge.get(value["version_id"], actors["admin"])
    assert error.value.code == "artifact_integrity"


def test_version_rows_are_immutable_and_build_is_audited(identities):
    store, actors, _ = identities
    knowledge = KnowledgeStore(store, java_path="/nonexistent/java")
    value = knowledge.build(original_knowledge(), actors["admin"])
    with store.transaction() as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE knowledge_versions SET build_status='checked' WHERE version_id=?", (value["version_id"],))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM knowledge_versions WHERE version_id=?", (value["version_id"],))
        with pytest.raises(sqlite3.IntegrityError, match="checked scoped publication"):
            db.execute("INSERT INTO knowledge_active VALUES(?,?,?,?)", ("PLANT-A", value["version_id"], 1, 0))
        audit = db.execute("SELECT * FROM audit WHERE event='knowledge_build'").fetchone()
        assert json.loads(audit["detail_json"])["content_hash"] == value["content_hash"]


def test_real_hermit_classifies_reloads_and_human_admin_publishes(identities):
    java = configured_java()
    import owlready2

    original_java_global = owlready2.JAVA_EXE
    store, actors, _ = identities
    knowledge = KnowledgeStore(store, java_path=java)
    value = knowledge.build(original_knowledge(), actors["analyst"])
    assert value["status"] == "checked", value["reasoner"]
    report = value["reasoner"]
    assert report["name"] == "HermiT"
    assert report["success"] and report["consistent"] and report["reload_verified"]
    assert report["unsatisfiable_classes"] == []
    assert len(report["hermit_jar_sha256"]) == len(report["java_binary_sha256"]) == 64
    assert report["hermit_exit_code"] == 0
    assert report["hermit_output_bytes"] > 0
    assert any(pair["subclass"].endswith("#rule_jam") and pair["superclass"].endswith("#EvidenceBoundLeaf")
               for pair in report["inferences"])
    assert owlready2.JAVA_EXE == original_java_global
    with pytest.raises(DomainError) as error:
        knowledge.active("PLANT-A", actors["operator"])
    assert error.value.code == "no_active_knowledge"
    wrong = publication(value).model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(DomainError) as error:
        knowledge.publish(wrong, actors["admin"])
    assert error.value.code == "knowledge_hash_mismatch"
    request = publication(value)
    published = knowledge.publish(request, actors["admin"])
    assert published == knowledge.publish(request, actors["admin"])
    active = knowledge.active("PLANT-A", actors["operator"])
    assert active["status"] == "published"
    assert active["version_id"] == value["version_id"]
    assert active["content_hash"] == value["content_hash"]
    assert active["draft"] == original_knowledge().draft.model_dump()
    assert active["active_revision"] == 1


def test_real_hermit_active_uses_caller_transaction_without_committing(identities, monkeypatch):
    store, actors, outsider = identities
    knowledge = KnowledgeStore(store, java_path=configured_java())
    value = knowledge.build(original_knowledge(), actors["admin"])
    assert value["status"] == "checked", value["reasoner"]
    knowledge.publish(publication(value), actors["admin"])
    expected = knowledge.active("PLANT-A", actors["operator"])
    with store.transaction() as db:
        def forbidden_nested_transaction():
            pytest.fail("active(db=...) must not open another transaction")

        with monkeypatch.context() as patch:
            patch.setattr(store, "transaction", forbidden_nested_transaction)
            assert knowledge.active("PLANT-A", actors["operator"], db=db) == expected
            assert db.in_transaction
            with pytest.raises(DomainError) as error:
                knowledge.active("PLANT-A", outsider, db=db)
            assert error.value.code == "forbidden"
            # Uncommitted ACL changes must be visible; an unrelated connection would miss this.
            db.execute("UPDATE users SET acl_version=acl_version+1 WHERE actor_id='operator'")
            with pytest.raises(DomainError) as error:
                knowledge.active("PLANT-A", actors["operator"], db=db)
            assert error.value.code == "identity_changed"
            db.rollback()
    # The helper neither committed our permission change nor closed the caller connection.
    assert knowledge.active("PLANT-A", actors["operator"])["version_id"] == value["version_id"]


def test_real_hermit_pinned_old_version_survives_new_activation(identities):
    java = configured_java()
    store, actors, _ = identities
    knowledge = KnowledgeStore(store, java_path=java)
    first = knowledge.build(original_knowledge(), actors["admin"])
    assert first["status"] == "checked", first["reasoner"]
    receipt = knowledge.publish(publication(first), actors["admin"])
    changed = original_knowledge()
    changed.request_id = "second-original-version"
    changed.draft.title = "Human revised synthetic conveyor rule tree"
    second = knowledge.build(changed, actors["admin"])
    assert second["status"] == "checked", second["reasoner"]
    knowledge.publish(publication(second, "publish-second"), actors["admin"])
    active = knowledge.active("PLANT-A", actors["operator"])
    assert active["version_id"] == second["version_id"] and active["active_revision"] == 2
    pinned = knowledge.get(first["version_id"], actors["operator"])
    assert pinned["draft"] == first["draft"] and pinned["content_hash"] == first["content_hash"]
    assert pinned["status"] == "published" and not pinned["active"]
    # Replaying an earlier successful action must not silently roll back the active pointer.
    assert knowledge.publish(publication(first), actors["admin"]) == receipt
    assert knowledge.active("PLANT-A", actors["operator"])["version_id"] == second["version_id"]
    with pytest.raises(DomainError) as error:
        knowledge.publish(publication(second), actors["admin"])
    assert error.value.code == "idempotency_conflict"
    classified = first["artifacts"]["classified_owl"]
    (store.artifacts / (classified["sha256"] + ".json")).write_text("{}")
    with pytest.raises(DomainError) as error:
        knowledge.get(first["version_id"], actors["operator"])
    assert error.value.code == "artifact_integrity"
