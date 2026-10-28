"""Explicit original-fixture acceptance with real OWL reasoning, no production interfaces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from industrial_diagnosis.fixtures import original_knowledge
from industrial_diagnosis.knowledge import KnowledgeStore
from industrial_diagnosis.managed_provider import ManagedLocalProvider
from industrial_diagnosis.provider import SyntheticProvider
from industrial_diagnosis.reports import ReportService
from industrial_diagnosis.schemas import DiagnosisRequest, HumanAnswer, PublishRequest, digest
from industrial_diagnosis.service import DiagnosisService
from industrial_diagnosis.storage import Store
from industrial_diagnosis.telemetry import TelemetryStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--test-provider", action="store_true")
    parser.add_argument("--cases", nargs="+", default=["normal", "jam", "slip", "sensor_fault", "missing", "conflicting"])
    args = parser.parse_args()
    if args.test_provider == (args.runtime_config is not None):
        parser.error("Select exactly --test-provider or --runtime-config; never fallback")
    if args.data_root.exists() or args.data_root.is_symlink():
        parser.error("Acceptance requires a new data root; no overwrite")
    if not set(args.cases) <= {"normal", "jam", "slip", "sensor_fault", "missing", "conflicting"}:
        parser.error("Unknown synthetic fixture")
    args.data_root.mkdir(mode=0o700, parents=True)
    result = {"status": "running", "synthetic_data": True,
              "provider": "TEST-only" if args.test_provider else "real-local-Qwen",
              "reasoner": "real-HermiT-required", "cases": []}
    output = args.data_root / "acceptance.json"
    started = time.monotonic()
    def save():
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    save()
    try:
        store = Store(args.data_root / "data")
        actor = store.authenticate(store.provision_user("acceptance-admin", "admin", ["PLANT-A"]))
        telemetry = TelemetryStore(store)
        listing = telemetry.seed()
        knowledge = KnowledgeStore(store, java_path=args.java)
        built = knowledge.build(original_knowledge(), actor)
        result["knowledge"] = {key: built[key] for key in ("version_id", "content_hash", "status", "reasoner")}
        save()
        assert built["status"] == "checked", "Actual HermiT build did not pass"
        knowledge.publish(PublishRequest(action_id="acceptance-publication", version_id=built["version_id"],
            content_hash=built["content_hash"], review_note="Reviewed ORIGINAL SYNTHETIC fixture sources, predicates and actual reasoner results for test use only."), actor)
        provider = SyntheticProvider() if args.test_provider else ManagedLocalProvider(args.runtime_config)
        result["provider_fingerprint"] = provider.fingerprint
        service = DiagnosisService(store, knowledge, telemetry, provider)
        report_service = ReportService(store, service, provider)
        for case in args.cases:
            event = listing["cases"][case]
            request = DiagnosisRequest(request_id="acceptance-" + case,
                **{k: event[k] for k in ("plant_id", "device_id", "event_id", "window", "complaint")})
            task = service.create(request, actor)
            entry = {"fixture": case, "task_id": task["task_id"], "status": "running", "human_fixture_answers": []}
            result["cases"].append(entry)
            save()
            task = service.run(task["task_id"], actor)
            for turn in range(3):
                if task["status"] != "waiting_human":
                    break
                question = task["pending_human"]
                cid = question["condition_id"]
                # Explicit test observations, supplied only through the human-answer boundary.
                # No expected outcome or fixture name enters the model context.
                answer = {
                    "jam": {"obstruction_confirmed": True},
                    "slip": {"traction_loss_confirmed": True},
                    "sensor_fault": {"independent_measurement_confirmed": True},
                    "conflicting": {"obstruction_confirmed": True, "traction_loss_confirmed": True},
                }.get(case, {}).get(cid, False)
                entry["human_fixture_answers"].append({"condition_id": cid, "answer": answer,
                    "source": "explicit original synthetic test observation, not live human evidence"})
                service.answer(task["task_id"], HumanAnswer(action_id=f"answer-{case}-{turn}",
                    question_id=question["question_id"], answer=answer,
                    observation_note="ORIGINAL SYNTHETIC acceptance observation supplied explicitly by the test fixture; no physical inspection performed."), actor)
                task = service.run(task["task_id"], actor)
            entry.update({key: task[key] for key in ("status", "calls", "tools", "tokens", "elapsed_s", "error", "limits", "compaction_count")})
            entry["diagnosis"] = task["submission"]["node_id"] if task["submission"] else None
            entry["evidence_states"] = {e["condition_id"]: e["state"] for e in task["evidence"]}
            entry["model_calls"] = service.calls(task["task_id"], actor)
            entry["tool_order"] = [json.loads(a["detail_json"])["tool"] for a in service.audit(task["task_id"], actor) if a["event"] == "tool_completed"]
            save()
            if case in {"missing", "conflicting"}:
                expected_code = "missing_signals" if case == "missing" else "conflicting_signals"
                assert task["status"] == "escalated" and task["submission"] is None
                assert task["error"]["code"] == expected_code, "Unrelated provider or budget failure is not negative-case success"
                assert all(call["status"] == "completed" for call in entry["model_calls"])
                witness = task["error"]["witness"]
                issued = {e["evidence_id"]: e for e in task["evidence"]}
                assert witness["evidence_ids"] and set(witness["evidence_ids"]) <= issued.keys()
                assert set(witness["condition_ids"]) == {issued[eid]["condition_id"] for eid in witness["evidence_ids"]}
                if case == "missing":
                    assert all(issued[eid]["source"] == "signal" and issued[eid]["state"] == "missing"
                               and issued[eid]["details"].get("reason") for eid in witness["evidence_ids"])
                else:
                    draft = built["draft"]
                    referenced = set(draft["global_required"])
                    for node in draft["nodes"]:
                        referenced.update(node["conditions"] + node["mandatory"])
                    expected_signals = {c["condition_id"] for c in draft["conditions"]
                                        if c["source"] == "signal" and c["condition_id"] in referenced}
                    assert set(witness["condition_ids"]) == expected_signals
                    assert {e["condition_id"] for e in task["evidence"] if e["source"] == "signal"} == expected_signals
                    assert all(e["state"] != "missing" for e in task["evidence"] if e["source"] == "signal")
                    assert witness["compatible_nodes"] == []
                    assert all(entry["evidence_states"][cid] == "supported"
                               for cid in ("current_high", "belt_low", "traction_difference"))
                    assert entry["evidence_states"]["current_nominal"] == "refuted"
                assert service.run(task["task_id"], actor) == task, "Terminal replay must preserve counters and outcome"
            else:
                assert task["status"] == "submitted", f"{case}: {task['status']} / {task['error']}"
                assert task["submission"]["node_id"] == case, "Independent synthetic oracle mismatch"
                frozen_hash = digest(task["submission"])
                report = report_service.generate(task["task_id"], "report-" + case, actor)
                entry["report"] = {key: report[key] for key in ("report_id", "status", "content_hash")}
                assert digest(service.get(task["task_id"], actor)["submission"]) == frozen_hash
                if not args.test_provider:
                    assert report["status"] == "completed", "Real post-END narrative did not complete"
            entry["checks_passed"] = True
            save()
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        raise
    finally:
        save()
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
