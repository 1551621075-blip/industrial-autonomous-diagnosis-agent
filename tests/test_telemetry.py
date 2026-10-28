from __future__ import annotations

import copy
import json
import math
import sqlite3

import pytest
from pydantic import ValidationError

from industrial_diagnosis.schemas import Condition, DomainError, Window, canonical, digest
from industrial_diagnosis.storage import Store
from industrial_diagnosis.telemetry import MAX_EVENT_POINTS, TelemetryStore, verify


@pytest.fixture
def telemetry(tmp_path):
    store = Store(tmp_path / "synthetic-only")
    token = store.provision_user("test-operator", "operator", ["PLANT-A"])
    actor = store.authenticate(token)
    telemetry = TelemetryStore(store)
    return telemetry, actor, telemetry.seed()


def snapshot_for(telemetry, case="normal"):
    service, actor, listing = telemetry
    case_metadata = listing["cases"][case]
    return service.event(
        case_metadata["plant_id"], case_metadata["device_id"], case_metadata["event_id"],
        Window(**case_metadata["window"]), actor,
    )


def condition(**changes):
    args = {
        "condition_id": "C-01", "label": "Motor load threshold", "source": "signal",
        "signal": "motor_current", "statistic": "mean", "operator": "gt",
        "threshold": 10.0, "unit": "A",
    }
    return Condition(**(args | changes))


def rehash(snapshot):
    snapshot["data_hash"] = digest({k: v for k, v in snapshot.items() if k != "data_hash"})
    return snapshot


def constant_snapshot(telemetry, value=5.0):
    snapshot = snapshot_for(telemetry)
    for point in snapshot["signals"]["motor_current"]:
        point["value"] = value
    return rehash(snapshot)


def assert_missing(result, reason=None):
    assert result["state"] == "missing"
    assert result["details"]["computed"] is None
    assert isinstance(result["details"]["n"], int)
    if reason is not None:
        assert result["details"]["reason"] == reason


def test_seed_lists_cases_without_leaking_oracle_into_snapshots(telemetry):
    service, _, listing = telemetry
    assert set(listing["cases"]) == {
        "jam", "slip", "sensor_fault", "normal", "missing", "conflicting",
    }
    assert set(listing["isolation_cases"]) == {"other_device", "other_plant"}
    for name, metadata in listing["cases"].items():
        snapshot = snapshot_for(telemetry, name)
        assert snapshot["synthetic"] is True
        assert snapshot["provenance"]["production_data"] is False
        assert "mathematically generated" in snapshot["provenance"]["source"]
        assert snapshot["event_id"].startswith("EVT-")
        assert snapshot["title"] == metadata["title"]
        assert snapshot["complaint"] == metadata["complaint"]
        assert not {"case", "fixture_expected", "diagnosis", "expected"} & snapshot.keys()
        assert name not in snapshot["event_id"]
        for text in (snapshot["title"], snapshot["complaint"]):
            assert not any(label in text for label in listing["cases"])
    with service.store.connect() as db:
        metadata = db.execute("SELECT metadata_json FROM telemetry_events").fetchall()
        assert len(metadata) == 8
        assert all("fixture_expected" not in row[0] and '"case"' not in row[0] for row in metadata)


def test_demo_cases_flattens_existing_listing_without_reseed_or_raw_samples(telemetry, monkeypatch):
    service, actor, listing = telemetry

    def forbidden_seed(*args, **kwargs):
        pytest.fail("demo_cases must never seed or overwrite telemetry")

    monkeypatch.setattr(service, "seed", forbidden_seed)
    before = snapshot_for(telemetry)
    actual = service.demo_cases(actor)
    expected = [
        {"case_label": label, **metadata, "synthetic": True}
        for label, metadata in (listing["cases"] | listing["isolation_cases"]).items()
        if metadata["plant_id"] in actor.plants
    ]
    assert actual == sorted(expected, key=lambda case: (
        case["plant_id"], case["device_id"], case["event_id"], case["case_label"],
    ))
    assert len(actual) == 7
    assert all(set(case) == {"case_label", "plant_id", "device_id", "event_id", "window",
                             "title", "complaint", "synthetic"} for case in actual)
    assert snapshot_for(telemetry) == before
    assert "case_label" not in snapshot_for(telemetry)
    actual[0]["window"]["end_ms"] = 999
    assert service.demo_cases(actor)[0]["window"]["end_ms"] == 30_000


@pytest.mark.parametrize(("plants", "expected_count"), [
    (["PLANT-B"], 1), (["PLANT-A", "PLANT-B"], 8), (["PLANT-C"], 0),
])
def test_demo_cases_filters_nested_case_plant_not_seed_plant(telemetry, plants, expected_count):
    service, _, _ = telemetry
    token = service.store.provision_user("demo-reader", "analyst", plants)
    actor = service.store.authenticate(token)
    result = service.demo_cases(actor)
    assert len(result) == expected_count
    assert all(case["plant_id"] in plants for case in result)
    if plants == ["PLANT-B"]:
        assert result[0]["case_label"] == "other_plant"
        assert result[0]["event_id"] == "EVT-008"


@pytest.mark.parametrize("enabled", [True, False])
def test_demo_cases_denies_stale_or_revoked_identity(telemetry, enabled):
    service, actor, _ = telemetry
    admin_token = service.store.provision_user("demo-admin", "admin", ["PLANT-A", "PLANT-B"])
    admin = service.store.authenticate(admin_token)
    service.store.set_user(actor.actor_id, admin, enabled=enabled, plants=["PLANT-B"], role="operator")
    with pytest.raises(DomainError) as error:
        service.demo_cases(actor)
    assert error.value.code == "identity_changed"
    assert error.value.status == 403


def test_demo_cases_denies_forged_plant_scope(telemetry):
    service, actor, _ = telemetry
    forged = actor.model_copy(update={"plants": ["PLANT-A", "PLANT-B"]})
    with pytest.raises(DomainError) as error:
        service.demo_cases(forged)
    assert error.value.code == "identity_changed"


def test_demo_cases_empty_store_does_not_seed(tmp_path):
    store = Store(tmp_path / "unseeded")
    actor = store.authenticate(store.provision_user("empty-reader", "operator", ["PLANT-A"]))
    service = TelemetryStore(store)
    assert service.demo_cases(actor) == []
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM telemetry_seeds").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM telemetry_samples").fetchone()[0] == 0


def test_demo_cases_survives_reopening_store(telemetry):
    service, actor, _ = telemetry
    reopened = TelemetryStore(Store(service.store.root))
    assert reopened.demo_cases(actor) == service.demo_cases(actor)


def test_duplicate_seed_refuses_without_modifying_stored_samples(telemetry):
    service, _, _ = telemetry
    before = snapshot_for(telemetry)
    with pytest.raises(DomainError) as error:
        service.seed()
    assert error.value.code == "telemetry_seed_exists"
    assert error.value.status == 409
    assert snapshot_for(telemetry) == before
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0] == 8


@pytest.mark.parametrize("plant", ["", "../escape", "PLANT A", "x" * 81, None])
def test_invalid_seed_plant_rejected(telemetry, plant):
    with pytest.raises(DomainError, match="Invalid plant"):
        telemetry[0].seed(plant)


def test_seed_is_deterministic_across_independent_stores(telemetry, tmp_path):
    service = TelemetryStore(Store(tmp_path / "another-synthetic-store"))
    token = service.store.provision_user("second-reader", "analyst", ["PLANT-A"])
    second = (service, service.store.authenticate(token), service.seed())
    assert snapshot_for(second, "jam") == snapshot_for(telemetry, "jam")


def test_generation_is_time_varying_coupled_and_changes_computation(telemetry):
    normal = snapshot_for(telemetry)
    jam = snapshot_for(telemetry, "jam")
    for name in ("motor_current", "belt_speed", "roller_speed", "encoder_speed"):
        assert len({p["value"] for p in normal["signals"][name]}) > 10
        assert [p["timestamp_ms"] for p in normal["signals"][name]] == list(range(0, 30_000, 1000))
    assert verify(normal, condition())["details"]["computed"] == pytest.approx(4)
    assert verify(jam, condition())["details"]["computed"] == pytest.approx(14)
    assert verify(normal, condition())["state"] == "refuted"
    assert verify(jam, condition())["state"] == "supported"
    changed = copy.deepcopy(normal)
    for point in changed["signals"]["motor_current"]:
        point["value"] += 9
    changed = rehash(changed)
    assert changed["event_id"] == normal["event_id"]
    assert verify(changed, condition())["details"]["computed"] == pytest.approx(13)
    assert verify(changed, condition())["state"] == "supported"


# This numerical oracle is test-only. Production telemetry stores no expected diagnoses.
@pytest.mark.parametrize(("case", "current", "belt", "roller", "encoder"), [
    ("jam", 14, 0.03, 0.06, 0.03),
    ("slip", 6, 0.45, 1.5, 0.45),
    ("sensor_fault", 4, 1.0, 1.0, 2.0),
    ("normal", 4, 1.0, 1.0, 1.0),
    ("conflicting", 14, 0.0375, 1.5, 0.0375),
])
def test_case_values_are_measured_not_labels(telemetry, case, current, belt, roller, encoder):
    snapshot = snapshot_for(telemetry, case)
    for signal, expected, unit in (
        ("motor_current", current, "A"), ("belt_speed", belt, "m/s"),
        ("roller_speed", roller, "m/s"), ("encoder_speed", encoder, "m/s"),
    ):
        result = verify(snapshot, condition(signal=signal, unit=unit))
        assert result["details"]["computed"] == pytest.approx(expected)
        assert result["details"]["n"] == 30
    assert "diagnosis" not in result


def test_conflicting_signals_are_reported_independently(telemetry):
    snapshot = snapshot_for(telemetry, "conflicting")
    assert verify(snapshot, condition())["state"] == "supported"
    assert verify(snapshot, condition(signal="belt_speed", operator="lt", threshold=0.1,
                                      unit="m/s"))["state"] == "supported"
    separation = condition(signal="roller_speed", other_signal="belt_speed",
                           statistic="difference_mean", unit="m/s", threshold=0.5)
    assert verify(snapshot, separation)["state"] == "supported"


@pytest.mark.parametrize(("operator", "threshold", "upper", "state"), [
    ("gt", 5, None, "refuted"), ("gt", 4.999, None, "supported"),
    ("lt", 5, None, "refuted"), ("lt", 5.001, None, "supported"),
    ("eq", 5, None, "supported"), ("eq", 5.000000001, None, "refuted"),
    ("between", 5, 6, "supported"), ("between", 4, 5, "supported"),
    ("between", 5, 5, "supported"), ("between", 5.001, 6, "refuted"),
])
def test_numerical_operator_boundaries(telemetry, operator, threshold, upper, state):
    result = verify(constant_snapshot(telemetry), condition(
        operator=operator, threshold=threshold, upper=upper,
    ))
    assert result["state"] == state
    assert result["details"]["computed"] == 5


@pytest.mark.parametrize("statistic", ["mean", "max", "min"])
def test_statistics_use_actual_sample_values(telemetry, statistic):
    snapshot = snapshot_for(telemetry)
    values = [point["value"] for point in snapshot["signals"]["motor_current"]]
    expected = {"mean": math.fsum(values) / len(values), "max": max(values), "min": min(values)}
    result = verify(snapshot, condition(statistic=statistic, operator="eq", threshold=expected[statistic]))
    assert result["details"]["computed"] == expected[statistic]
    assert result["state"] == "supported"


def test_difference_mean_signed_and_exactly_paired(telemetry):
    snapshot = snapshot_for(telemetry, "slip")
    first = condition(signal="roller_speed", other_signal="belt_speed", statistic="difference_mean",
                      unit="m/s", operator="gt", threshold=1)
    result = verify(snapshot, first)
    assert result["details"]["computed"] == pytest.approx(1.05)
    assert result["details"]["n"] == 30
    assert result["state"] == "supported"
    reverse = first.model_copy(update={"signal": "belt_speed", "other_signal": "roller_speed"})
    assert verify(snapshot, reverse)["details"]["computed"] == pytest.approx(-1.05)
    assert verify(snapshot, reverse)["state"] == "refuted"
    snapshot["signals"]["belt_speed"][10]["timestamp_ms"] += 1
    assert_missing(verify(rehash(snapshot), first), "unpaired_timestamps")


def test_difference_rejects_partial_pairing_even_when_both_counts_suffice(telemetry):
    snapshot = snapshot_for(telemetry)
    snapshot["signals"]["belt_speed"].pop(10)
    predicate = condition(signal="roller_speed", other_signal="belt_speed", statistic="difference_mean",
                          unit="m/s", min_samples=20, max_gap_ms=2500)
    assert_missing(verify(rehash(snapshot), predicate), "unpaired_timestamps")


def test_subset_window_is_left_closed_right_open(telemetry):
    service, actor, listing = telemetry
    event = listing["cases"]["normal"]
    narrow = service.event("PLANT-A", event["device_id"], event["event_id"],
                           Window(start_ms=5000, end_ms=10_000), actor)
    assert narrow["window"] == {"start_ms": 5000, "end_ms": 10_000}
    assert narrow["event_window"] == {"start_ms": 0, "end_ms": 30_000}
    assert [p["timestamp_ms"] for p in narrow["signals"]["motor_current"]] == [
        5000, 6000, 7000, 8000, 9000,
    ]
    full = snapshot_for(telemetry)
    assert full["event_data_hash"] == narrow["event_data_hash"]
    assert full["data_hash"] != narrow["data_hash"]
    assert_missing(verify(narrow, condition()), "insufficient_samples")
    measured = verify(narrow, condition(min_samples=5))
    assert measured["details"]["n"] == 5
    assert measured["details"]["computed"] == pytest.approx(math.fsum(
        p["value"] for p in full["signals"]["motor_current"] if 5000 <= p["timestamp_ms"] < 10_000
    ) / 5)


@pytest.mark.parametrize(("start", "end"), [(0, 30_001), (30_000, 31_000), (1, 3_600_001)])
def test_outside_event_window_rejected(telemetry, start, end):
    service, actor, _ = telemetry
    with pytest.raises(DomainError) as error:
        service.event("PLANT-A", "CONVEYOR-01", "EVT-001", Window(start_ms=start, end_ms=end), actor)
    assert error.value.code == "telemetry_window_mismatch"


def test_oversized_windows_rejected_even_if_model_construct_bypassed_validation(telemetry):
    service, actor, _ = telemetry
    with pytest.raises(ValidationError):
        Window(start_ms=0, end_ms=3_600_001)
    bad_window = Window.model_construct(start_ms=0, end_ms=3_600_001)
    with pytest.raises(DomainError) as error:
        service.event("PLANT-A", "CONVEYOR-01", "EVT-001", bad_window, actor)
    assert error.value.code == "invalid_window"


@pytest.mark.parametrize(("plant", "device", "event", "code"), [
    ("PLANT-B", "CONVEYOR-01", "EVT-008", "forbidden"),
    ("PLANT-A", "CONVEYOR-02", "EVT-001", "telemetry_not_found"),
    ("PLANT-A", "CONVEYOR-01", "EVT-007", "telemetry_not_found"),
    ("PLANT-A", "CONVEYOR-01", "EVT-008", "telemetry_not_found"),
    ("PLANT-A", "CONVEYOR-01", "EVT-999", "telemetry_not_found"),
    ("PLANT-A", "CONVEYOR-01' OR 1=1--", "EVT-001", "telemetry_not_found"),
])
def test_cross_plant_device_event_and_sql_input_do_not_mix(telemetry, plant, device, event, code):
    service, actor, _ = telemetry
    with pytest.raises(DomainError) as error:
        service.event(plant, device, event, Window(start_ms=0, end_ms=30_000), actor)
    assert error.value.code == code


def test_isolation_fixtures_can_only_be_read_in_their_own_scope(telemetry):
    service, actor, listing = telemetry
    token = service.store.provision_user("other-operator", "operator", ["PLANT-B"])
    other_actor = service.store.authenticate(token)
    for name, metadata in listing["isolation_cases"].items():
        reader = actor if name == "other_device" else other_actor
        snapshot = service.event(metadata["plant_id"], metadata["device_id"], metadata["event_id"],
                                 Window(**metadata["window"]), reader)
        assert snapshot["plant_id"] == metadata["plant_id"]
        assert snapshot["device_id"] == metadata["device_id"]
        assert snapshot["event_id"] == metadata["event_id"]


def test_revoked_actor_cannot_read_even_with_old_actor_object(telemetry):
    service, actor, _ = telemetry
    token = service.store.provision_user("test-admin", "admin", ["PLANT-A"])
    admin = service.store.authenticate(token)
    service.store.set_user(actor.actor_id, admin, enabled=False, plants=["PLANT-A"], role="operator")
    with pytest.raises(DomainError) as error:
        snapshot_for(telemetry)
    assert error.value.code == "identity_changed"


def test_missing_fixture_reports_gap_and_absent_sensor(telemetry):
    snapshot = snapshot_for(telemetry, "missing")
    gap = verify(snapshot, condition())
    assert_missing(gap, "coverage_gap")
    assert gap["details"]["n"] == 25
    assert gap["details"]["max_gap_ms"] == 6000
    assert_missing(verify(snapshot, condition(signal="encoder_speed", unit="m/s")), "signal_absent")
    assert verify(snapshot, condition(signal="belt_speed", unit="m/s", operator="gt",
                                      threshold=0.8))["state"] == "supported"


@pytest.mark.parametrize(("drop", "reason"), [
    ([0, 1], "coverage_gap"), ([28, 29], "coverage_gap"), ([10], "coverage_gap"),
    (list(range(11)), "insufficient_samples"),
])
def test_count_and_start_end_interior_gap_coverage(telemetry, drop, reason):
    snapshot = snapshot_for(telemetry)
    snapshot["signals"]["motor_current"] = [
        p for i, p in enumerate(snapshot["signals"]["motor_current"]) if i not in drop
    ]
    assert_missing(verify(rehash(snapshot), condition()), reason)


def test_gap_age_and_sample_count_exact_boundaries(telemetry):
    snapshot = snapshot_for(telemetry)
    exact = condition(min_samples=30, max_gap_ms=1000, max_age_ms=1000)
    assert verify(snapshot, exact)["state"] == "refuted"
    assert_missing(verify(snapshot, exact.model_copy(update={"min_samples": 31})), "insufficient_samples")
    assert_missing(verify(snapshot, exact.model_copy(update={"max_gap_ms": 999})), "coverage_gap")
    assert_missing(verify(snapshot, exact.model_copy(update={"max_age_ms": 999})), "stale_samples")
    assert_missing(verify(snapshot, exact.model_copy(update={"max_age_ms": 0})), "stale_samples")


@pytest.mark.parametrize("mutation", ["duplicate", "nonmonotonic", "future", "before", "float", "bool"])
def test_timestamp_corruption_fails_missing(telemetry, mutation):
    snapshot = snapshot_for(telemetry)
    points = snapshot["signals"]["motor_current"]
    reasons = {
        "duplicate": "nonmonotonic_timestamp", "nonmonotonic": "nonmonotonic_timestamp",
        "future": "sample_outside_window", "before": "sample_outside_window",
        "float": "invalid_timestamp", "bool": "invalid_timestamp",
    }
    if mutation == "duplicate":
        points[11]["timestamp_ms"] = points[10]["timestamp_ms"]
    elif mutation == "nonmonotonic":
        points[10], points[11] = points[11], points[10]
    elif mutation == "future":
        points[-1]["timestamp_ms"] = snapshot["window"]["end_ms"]
    elif mutation == "before":
        snapshot["window"]["start_ms"] = 1
    elif mutation == "float":
        points[10]["timestamp_ms"] = 10_000.0
    else:
        points[0]["timestamp_ms"] = False
    assert_missing(verify(rehash(snapshot), condition()), reasons[mutation])


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("quality", "bad", "invalid_sensor"), ("quality", "uncertain", "invalid_sensor"),
    ("quality", None, "invalid_sensor"), ("sensor_valid", False, "invalid_sensor"),
    ("sensor_valid", 1, "invalid_sensor"), ("sensor_valid", None, "invalid_sensor"),
    ("value", None, "invalid_value"), ("value", "4.0", "invalid_value"),
    ("value", True, "invalid_value"), ("unit", "mA", "unit_mismatch"),
])
def test_sample_quality_validity_values_units_fail_closed(telemetry, field, value, reason):
    snapshot = snapshot_for(telemetry)
    snapshot["signals"]["motor_current"][15][field] = value
    assert_missing(verify(rehash(snapshot), condition()), reason)


@pytest.mark.parametrize("field", ["timestamp_ms", "value", "quality", "sensor_valid", "unit"])
def test_absent_sample_fields_fail_missing(telemetry, field):
    snapshot = snapshot_for(telemetry)
    del snapshot["signals"]["motor_current"][5][field]
    assert_missing(verify(rehash(snapshot), condition()))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_samples_fail_missing_without_exception(telemetry, value):
    snapshot = snapshot_for(telemetry)
    snapshot["signals"]["motor_current"][5]["value"] = value
    assert_missing(verify(snapshot, condition()), "invalid_snapshot")


def test_finite_inputs_with_overflowing_statistic_fail_missing(telemetry):
    snapshot = constant_snapshot(telemetry, value=1e308)
    assert_missing(verify(snapshot, condition()), "nonfinite_computation")


def test_units_are_validated_for_condition_metadata_and_both_difference_signals(telemetry):
    snapshot = snapshot_for(telemetry)
    assert_missing(verify(snapshot, condition(unit="mA")), "unit_mismatch")
    snapshot["units"]["motor_current"] = "mA"
    assert_missing(verify(rehash(snapshot), condition()), "unit_mismatch")
    snapshot = snapshot_for(telemetry)
    mixed = condition(statistic="difference_mean", other_signal="belt_speed")
    assert_missing(verify(snapshot, mixed), "unit_mismatch")
    pair = condition(signal="roller_speed", other_signal="belt_speed", statistic="difference_mean",
                     unit="m/s")
    snapshot["signals"]["belt_speed"][10]["unit"] = "mm/s"
    assert_missing(verify(rehash(snapshot), pair), "unit_mismatch")


def test_drive_enabled_uses_numeric_zero_one_and_bool_unit(telemetry):
    snapshot = snapshot_for(telemetry)
    predicate = condition(signal="drive_enabled", unit="bool", operator="eq", threshold=1)
    assert verify(snapshot, predicate)["state"] == "supported"
    snapshot["signals"]["drive_enabled"][5]["value"] = 2
    assert_missing(verify(rehash(snapshot), predicate), "invalid_value")


def test_knowledge_fixture_units_and_global_requirements_match_telemetry(telemetry):
    from industrial_diagnosis.fixtures import original_knowledge
    from industrial_diagnosis.telemetry import SIGNAL_UNITS

    draft = original_knowledge().draft
    conditions = {predicate.condition_id: predicate for predicate in draft.conditions}
    for predicate in draft.conditions:
        if predicate.source == "signal":
            assert predicate.unit == SIGNAL_UNITS[predicate.signal]
            if predicate.other_signal is not None:
                assert predicate.unit == SIGNAL_UNITS[predicate.other_signal]
    assert draft.global_required
    service, _, listing = telemetry
    token = service.store.provision_user("global-checker", "operator", ["PLANT-A", "PLANT-B"])
    reader = service.store.authenticate(token)
    for metadata in (listing["cases"] | listing["isolation_cases"]).values():
        snapshot = service.event(metadata["plant_id"], metadata["device_id"], metadata["event_id"],
                                 Window(**metadata["window"]), reader)
        for cid in draft.global_required:
            result = service.verify(snapshot, conditions[cid])
            assert result["state"] == "supported"
            assert result["details"]["computed"] == 1
            assert result["details"]["n"] == 30
            assert result["details"]["unit"] == "bool"


def test_human_source_is_explicitly_rejected(telemetry):
    human = Condition(condition_id="H-01", label="Mechanical inspection", source="human",
                      human_key="inspection", operator="eq", threshold=True, unit="bool")
    with pytest.raises(DomainError) as error:
        verify(snapshot_for(telemetry), human)
    assert error.value.code == "signal_condition_required"


def test_snapshot_hash_is_actual_canonical_contents_and_metadata(telemetry):
    snapshot = snapshot_for(telemetry)
    expected = digest({k: v for k, v in snapshot.items() if k != "data_hash"})
    assert snapshot["data_hash"] == expected
    event_payload = {k: v for k, v in snapshot.items() if k not in ("window", "data_hash", "event_data_hash")}
    assert snapshot["event_data_hash"] == digest(event_payload)
    assert json.loads(canonical(snapshot)) == snapshot
    assert verify(snapshot, condition()) == telemetry[0].verify(snapshot, condition())


@pytest.mark.parametrize("field", ["value", "plant_id", "device_id", "event_id", "title", "window"])
def test_changed_sample_or_metadata_without_rehash_is_missing(telemetry, field):
    snapshot = snapshot_for(telemetry)
    if field == "value":
        snapshot["signals"]["motor_current"][0]["value"] += 100
    elif field == "window":
        snapshot["window"]["end_ms"] -= 1000
    else:
        snapshot[field] += "-changed"
    assert_missing(verify(snapshot, condition()), "snapshot_hash_mismatch")
    assert snapshot_for(telemetry)["data_hash"] != digest({
        k: v for k, v in snapshot.items() if k != "data_hash"
    })


def test_provenance_cannot_be_omitted_even_with_new_hash(telemetry):
    snapshot = snapshot_for(telemetry)
    del snapshot["synthetic"]
    assert_missing(verify(rehash(snapshot), condition()), "invalid_provenance")


@pytest.mark.parametrize("snapshot", [None, [], {}, {"signals": []}, {"signals": {"motor_current": None}}])
def test_malformed_snapshots_fail_missing(telemetry, snapshot):
    assert_missing(verify(snapshot, condition()))


def test_snapshot_window_larger_than_hour_fails_missing(telemetry):
    snapshot = snapshot_for(telemetry)
    snapshot["window"]["end_ms"] = 3_600_001
    snapshot["event_window"]["end_ms"] = 3_600_001
    assert_missing(verify(rehash(snapshot), condition()), "invalid_snapshot")


def test_over_10000_points_are_rejected_before_hashing_or_computation(telemetry):
    snapshot = snapshot_for(telemetry)
    snapshot["signals"]["motor_current"] *= 334
    assert sum(map(len, snapshot["signals"].values())) > MAX_EVENT_POINTS
    assert_missing(verify(snapshot, condition()), "point_limit")


@pytest.mark.parametrize(("table", "operation"), [
    ("telemetry_events", "UPDATE telemetry_events SET data_hash='modified'"),
    ("telemetry_events", "DELETE FROM telemetry_events"),
    ("telemetry_samples", "UPDATE telemetry_samples SET value_json='999'"),
    ("telemetry_samples", "DELETE FROM telemetry_samples"),
    ("telemetry_seeds", "DELETE FROM telemetry_seeds"),
    ("telemetry_seeds", "UPDATE telemetry_seeds SET listing_json='{}'"),
    ("telemetry_samples", "INSERT OR REPLACE INTO telemetry_samples SELECT * FROM telemetry_samples"),
    ("telemetry_events", "INSERT OR REPLACE INTO telemetry_events SELECT * FROM telemetry_events"),
    ("telemetry_seeds", "INSERT OR REPLACE INTO telemetry_seeds SELECT * FROM telemetry_seeds"),
])
def test_database_rejects_mutating_or_replacing_sealed_contents(telemetry, table, operation):
    service, _, _ = telemetry
    before = snapshot_for(telemetry)
    with pytest.raises(sqlite3.IntegrityError), service.store.transaction() as db:
        db.execute(operation)
    assert snapshot_for(telemetry) == before


def test_appending_samples_to_sealed_event_is_refused(telemetry):
    service, _, _ = telemetry
    with pytest.raises(sqlite3.IntegrityError), service.store.transaction() as db:
        db.execute("INSERT INTO telemetry_samples VALUES(?,?,?,?,?,?,?,?,?,?)", (
            "PLANT-A", "CONVEYOR-01", "EVT-001", "motor_current", 99, 29_500, "22", "A", "good", 1,
        ))


def test_storage_corruption_is_detected_if_immutable_trigger_is_bypassed(telemetry):
    service, _, _ = telemetry
    # Deliberate tampering is confined to this test's own temporary synthetic database.
    with service.store.transaction() as db:
        db.execute("DROP TRIGGER telemetry_samples_no_update")
        db.execute("UPDATE telemetry_samples SET value_json='999' WHERE event_id='EVT-004'")
    with pytest.raises(DomainError) as error:
        snapshot_for(telemetry)
    assert error.value.code == "telemetry_integrity"


def test_raw_read_rejects_over_limit_before_fetching_samples(telemetry):
    service, _, _ = telemetry
    with service.store.transaction() as db:
        db.execute("DROP TRIGGER telemetry_samples_sealed")
        db.executemany("INSERT INTO telemetry_samples VALUES(?,?,?,?,?,?,?,?,?,?)", [
            ("PLANT-A", "CONVEYOR-01", "EVT-004", "motor_current", i, i, "4", "A", "good", 1)
            for i in range(30, MAX_EVENT_POINTS + 1)
        ])
    with pytest.raises(DomainError) as error:
        snapshot_for(telemetry)
    assert error.value.code == "telemetry_integrity"


def test_raw_read_has_explicit_row_limit(telemetry, monkeypatch):
    from contextlib import contextmanager

    service, _, _ = telemetry
    original_connect = service.store.connect
    statements = []

    @contextmanager
    def traced_connect():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(service.store, "connect", traced_connect)
    snapshot_for(telemetry)
    reads = [sql for sql in statements if "SELECT signal,timestamp_ms" in sql]
    assert len(reads) == 1
    assert f"LIMIT {MAX_EVENT_POINTS}" in reads[0]
