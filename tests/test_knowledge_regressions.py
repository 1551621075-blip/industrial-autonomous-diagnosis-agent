from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from industrial_diagnosis.fixtures import original_knowledge, original_reports
from industrial_diagnosis.knowledge import (
    KnowledgeStore,
    _check_local_owl,
    _classify,
    draft_from_reports,
    draft_from_yaml,
    export_owl,
)
from industrial_diagnosis.schemas import DomainError, PublishRequest
from industrial_diagnosis.storage import Store


def _java_or_skip():
    path = os.environ.get("INDUSTRIAL_DIAGNOSIS_JAVA") or os.environ.get("JAVA_PATH")
    if not path and os.environ.get("JAVA_HOME"):
        path = str(Path(os.environ["JAVA_HOME"]) / "bin" / "java")
    if not path:
        pytest.skip("Real HermiT integration requires an explicitly configured local JVM")
    return path


def _fake_java(tmp_path, body):
    path = tmp_path / "controlled-java-test-double"
    path.write_text(f"#!{sys.executable}\nimport sys, time\n" + body)
    path.chmod(0o700)
    return str(path)


@pytest.mark.parametrize("document", [
    "!!python/object/apply:os.system ['do-not-execute']",
    "a: &loop [*loop]",
    "a: 1\na: 2",
    "? [a, b]\n: value",
    "a: " + "[" * 40 + "1" + "]" * 40,
    "a: " + "x" * 256_001,
])
def test_safe_yaml_rejects_executable_recursive_duplicate_and_oversized_input(document):
    with pytest.raises(DomainError) as error:
        draft_from_yaml(document, original_reports())
    assert error.value.code == "invalid_draft"


def test_reports_cannot_disagree_or_smuggle_publishing_fields():
    reports = original_reports()
    document = yaml.safe_load(reports[1].text)
    document["knowledge"]["plant_id"] = "PLANT-B"
    reports[1].text = yaml.safe_dump(document)
    with pytest.raises(DomainError) as error:
        draft_from_reports(reports)
    assert error.value.code == "invalid_draft"
    document["knowledge"].pop("plant_id")
    document["knowledge"]["status"] = "published"
    reports[1].text = yaml.safe_dump(document)
    with pytest.raises(DomainError) as error:
        draft_from_reports(reports)
    assert error.value.code == "invalid_draft"


def test_model_source_mutation_does_not_change_original_provenance():
    reports = original_reports()
    initial = reports[0].model_dump()

    class LocalModel:
        local_only = True

        def extract(self, sources):
            sources[0].text = "Changed untrusted report content that is not original source material."
            return original_knowledge().draft.model_dump()

    draft_from_reports(reports, extractor=LocalModel())
    assert reports[0].model_dump() == initial


@pytest.mark.parametrize("payload", [
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:owl="http://www.w3.org/2002/07/owl#"><owl:Ontology rdf:about="urn:original:conveyor">'
    '<owl:imports rdf:resource="https://untrusted.invalid/ontology"/></owl:Ontology></rdf:RDF>',
    '<!DOCTYPE rdf:RDF [<!ENTITY remote SYSTEM "file:///do-not-read">]><rdf:RDF/>',
])
def test_owl_network_imports_and_external_entities_rejected_before_loader(payload):
    with pytest.raises(DomainError) as error:
        _check_local_owl(payload)
    assert error.value.code in {"ontology_import_forbidden", "invalid_ontology"}


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 301, True, "30"])
def test_reasoner_requires_finite_configuration(tmp_path, timeout):
    with pytest.raises(DomainError) as error:
        KnowledgeStore(Store(tmp_path / "state"), timeout_seconds=timeout)
    assert error.value.code == "invalid_reasoner_config"


def test_reasoner_requires_explicit_java_path_and_never_uses_ambient_path(tmp_path, monkeypatch):
    for key in ("INDUSTRIAL_DIAGNOSIS_JAVA", "JAVA_PATH", "JAVA_HOME"):
        monkeypatch.delenv(key, raising=False)
    store = Store(tmp_path / "state")
    assert KnowledgeStore(store).java_path is None
    with pytest.raises(DomainError) as error:
        KnowledgeStore(store, java_path="java")
    assert error.value.code == "invalid_java_config"
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "configured-jvm"))
    assert KnowledgeStore(store).java_path == str(tmp_path / "configured-jvm" / "bin" / "java")


def test_zero_exit_from_fake_java_is_not_reasoner_success(tmp_path):
    # Deliberate stub tests fail-closed behavior; this is NOT a successful HermiT test.
    java = _fake_java(tmp_path,
                      'if "-version" in sys.argv:\n    print(\'openjdk version "test-double"\')\n'
                      'else:\n    print("Syntax OK, but no classification occurred")\n')
    request = original_knowledge()
    report, classified = _classify(export_owl(request.draft), request.draft, java, 10)
    assert report["status"] == "reasoner_error", report
    assert report["success"] is False and classified is None
    assert report["consistent"] is None
    assert report["reload_verified"] is False


def test_missing_runtime_is_distinct_from_reasoning_timeout(tmp_path):
    java = _fake_java(tmp_path, 'print("No Java runtime installed")\nsys.exit(1)\n')
    request = original_knowledge()
    report, classified = _classify(export_owl(request.draft), request.draft, java, 10)
    assert report["status"] == "missing_jvm", report
    assert report["consistent"] is None and classified is None


def test_worker_timeout_is_finite_and_blocks_publication(tmp_path):
    java = _fake_java(tmp_path, 'time.sleep(60)\n')
    store = Store(tmp_path / "state")
    actor = store.authenticate(store.provision_user("admin", "admin", ["PLANT-A"]))
    knowledge = KnowledgeStore(store, java_path=java, timeout_seconds=0.5)
    started = time.monotonic()
    value = knowledge.build(original_knowledge(), actor)
    assert time.monotonic() - started < 5
    assert value["build_status"] == "blocked"
    assert value["reasoner"]["status"] == "timeout", value["reasoner"]
    with pytest.raises(DomainError) as error:
        knowledge.publish(PublishRequest(action_id="timeout-publish", version_id=value["version_id"],
                                         content_hash=value["content_hash"], review_note="Attempted human review."), actor)
    assert error.value.code == "knowledge_not_checked"


def test_worker_output_limit_is_distinct_and_bounded(tmp_path):
    java = _fake_java(tmp_path, 'sys.stdout.write("x" * 200000)\nsys.stdout.flush()\n')
    request = original_knowledge()
    report, classified = _classify(export_owl(request.draft), request.draft, java, 10)
    assert report["status"] == "output_limit", report
    assert report["success"] is False and classified is None


def test_no_owlready_global_java_race_between_workers(tmp_path):
    import owlready2

    java_before = owlready2.JAVA_EXE
    request = original_knowledge()
    text = export_owl(request.draft)
    fast = _fake_java(tmp_path, 'print("No runtime")\nsys.exit(1)\n')
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda path: _classify(text, request.draft, path, 10),
                                    [fast, "/nonexistent/other-java"]))
    assert all(result[0]["status"] == "missing_jvm" for result in results)
    assert owlready2.JAVA_EXE == java_before


def test_concurrent_builds_preserve_single_idempotent_version(tmp_path):
    store = Store(tmp_path / "state")
    actor = store.authenticate(store.provision_user("admin", "admin", ["PLANT-A"]))
    knowledge = KnowledgeStore(store, java_path="/nonexistent/java")
    with ThreadPoolExecutor(max_workers=2) as executor:
        values = list(executor.map(lambda _: knowledge.build(original_knowledge(), actor), range(2)))
    assert values[0]["version_id"] == values[1]["version_id"]
    assert len(knowledge.list_versions(actor)) == 1


def test_artifact_symlinks_cannot_bypass_knowledge_integrity(tmp_path):
    store = Store(tmp_path / "state")
    actor = store.authenticate(store.provision_user("admin", "admin", ["PLANT-A"]))
    knowledge = KnowledgeStore(store, java_path="/nonexistent/java")
    value = knowledge.build(original_knowledge(), actor)
    reference = value["artifacts"]["reports"]
    path = store.artifacts / (reference["sha256"] + ".json")
    outside = tmp_path / "external-original-reports.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(DomainError) as error:
        knowledge.get(value["version_id"], actor)
    assert error.value.code == "artifact_integrity"


def test_original_numeric_rules_align_with_synthetic_telemetry_units(tmp_path):
    from industrial_diagnosis.schemas import Window
    from industrial_diagnosis.telemetry import TelemetryStore

    store = Store(tmp_path / "state")
    actor = store.authenticate(store.provision_user("admin", "admin", ["PLANT-A"]))
    telemetry = TelemetryStore(store)
    listing = telemetry.seed()["cases"]
    draft = original_knowledge().draft
    conditions = {c.condition_id: c for c in draft.conditions}
    for diagnosis in ("jam", "slip", "sensor_fault", "normal"):
        case = listing[diagnosis]
        snapshot = telemetry.event(case["plant_id"], case["device_id"], case["event_id"],
                                   Window.model_validate(case["window"]), actor)
        node = next(n for n in draft.nodes if n.node_id == diagnosis)
        for condition_id in draft.global_required + node.conditions:
            evidence = telemetry.verify(snapshot, conditions[condition_id])
            assert evidence["state"] == "supported", (diagnosis, condition_id, evidence)
        for condition_id in node.mandatory:
            assert conditions[condition_id].source == "human"
            with pytest.raises(DomainError) as error:
                telemetry.verify(snapshot, conditions[condition_id])
            assert error.value.code == "signal_condition_required"


def test_real_hermit_distinguishes_inconsistency_from_unsatisfiable_named_class():
    java = _java_or_skip()
    request = original_knowledge()
    text = export_owl(request.draft)
    # TreeNode and Condition are disjoint. This dual-typed existing individual is inconsistent.
    contradictory = text.replace("</rdf:RDF>", '''
<rdf:Description rdf:about="urn:original:conveyor#node_jam">
  <rdf:type rdf:resource="urn:original:conveyor#Condition"/>
</rdf:Description>
</rdf:RDF>''')
    report, classified = _classify(contradictory, request.draft, java, 30)
    assert report["status"] == "inconsistent", report
    assert report["consistent"] is False and report["success"] is False and classified is None
    # With no individual in the impossible class, the ontology is consistent but not publishable.
    unsatisfiable = text.replace("</rdf:RDF>", '''
<owl:Class rdf:about="urn:original:conveyor#ImpossibleConditionRole">
  <rdfs:subClassOf rdf:resource="urn:original:conveyor#SignalCondition"/>
  <rdfs:subClassOf rdf:resource="urn:original:conveyor#HumanCondition"/>
</owl:Class>
</rdf:RDF>''')
    report, classified = _classify(unsatisfiable, request.draft, java, 30)
    assert report["status"] == "unsatisfiable", report
    assert report["consistent"] is True and report["success"] is False and classified is None
    assert "urn:original:conveyor#ImpossibleConditionRole" in report["unsatisfiable_classes"]
