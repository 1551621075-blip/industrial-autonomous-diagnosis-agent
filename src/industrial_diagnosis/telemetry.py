from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any

from pydantic import ValidationError

from .schemas import Actor, Condition, DomainError, Window, digest
from .storage import Store

MAX_EVENT_POINTS = 10_000
MAX_WINDOW_MS = 3_600_000
SIGNAL_UNITS = {
    "motor_current": "A",
    "belt_speed": "m/s",
    "roller_speed": "m/s",
    "encoder_speed": "m/s",
    "drive_enabled": "bool",
}
_SCHEMA = "synthetic-telemetry-v1"
_PROVENANCE = {
    "source": "mathematically generated synthetic conveyor telemetry",
    "generator": "coupled-conveyor-waves-v1",
    "production_data": False,
    "sample_period_ms": 1000,
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _window(value: Any) -> Window:
    try:
        checked = Window.model_validate(value.model_dump() if isinstance(value, Window) else value)
    except (ValidationError, TypeError, ValueError) as exc:
        raise DomainError("invalid_window", "A bounded, nonempty millisecond window is required") from exc
    if checked.end_ms - checked.start_ms > MAX_WINDOW_MS:
        raise DomainError("invalid_window", "Telemetry windows cannot exceed one hour")
    return checked


def _metadata(plant_id: str, device_id: str, event_id: str) -> dict:
    # These descriptions intentionally carry observations, not fixture diagnoses.
    return {
        "schema_version": _SCHEMA,
        "plant_id": plant_id,
        "device_id": device_id,
        "event_id": event_id,
        "event_window": {"start_ms": 0, "end_ms": 30_000},
        "title": "Synthetic conveyor operating interval",
        "complaint": "Review conveyor motion, motor load, and speed measurement consistency.",
        "synthetic": True,
        "provenance": dict(_PROVENANCE),
        "units": dict(SIGNAL_UNITS),
    }


def _generate(*, load: float, shaft_speed: float, transfer: float, encoder_gain: float) -> dict:
    """Generate coupled, time-varying signals; no verification or diagnosis is encoded here."""
    signals: dict[str, list[dict]] = {name: [] for name in SIGNAL_UNITS}
    for i in range(30):
        phase = 2 * math.pi * i / 30
        shaft = shaft_speed * (1 + 0.025 * math.sin(phase))
        belt = shaft * transfer + 0.002 * math.sin(2 * phase)
        values = {
            "motor_current": 2 + 8 * load + (0.15 + 0.3 * load) * math.cos(phase),
            "roller_speed": shaft,
            "belt_speed": belt,
            "encoder_speed": belt * encoder_gain + 0.001 * math.sin(3 * phase),
            "drive_enabled": 1.0,
        }
        for name, value in values.items():
            signals[name].append({
                "timestamp_ms": i * 1000,
                "value": round(value, 8),
                "unit": SIGNAL_UNITS[name],
                "quality": "good",
                "sensor_valid": True,
            })
    return signals


class TelemetryStore:
    """Local synthetic telemetry with sealed, append-only event contents.

    Only seed() writes samples. There is no model-facing ingestion, update, or SQL API.
    An event snapshot is a fresh dictionary whose hash pins its canonical contents;
    mutating that dictionary does not alter storage and invalidates its hash.
    """

    def __init__(self, store: Store):
        self.store = store
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS telemetry_seeds(
                    plant_id TEXT PRIMARY KEY, listing_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS telemetry_events(
                    plant_id TEXT NOT NULL, device_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, data_hash TEXT NOT NULL,
                    sample_count INTEGER NOT NULL CHECK(sample_count BETWEEN 0 AND 10000),
                    PRIMARY KEY(plant_id,device_id,event_id));
                CREATE TABLE IF NOT EXISTS telemetry_samples(
                    plant_id TEXT NOT NULL, device_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    signal TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL, value_json TEXT NOT NULL,
                    unit TEXT NOT NULL, quality TEXT NOT NULL, sensor_valid INTEGER NOT NULL,
                    PRIMARY KEY(plant_id,device_id,event_id,signal,ordinal),
                    FOREIGN KEY(plant_id,device_id,event_id)
                        REFERENCES telemetry_events(plant_id,device_id,event_id)
                        DEFERRABLE INITIALLY DEFERRED);
                CREATE TRIGGER IF NOT EXISTS telemetry_samples_sealed
                BEFORE INSERT ON telemetry_samples
                WHEN EXISTS(SELECT 1 FROM telemetry_events
                    WHERE plant_id=NEW.plant_id AND device_id=NEW.device_id
                    AND event_id=NEW.event_id)
                BEGIN SELECT RAISE(ABORT, 'Telemetry event is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS telemetry_events_no_replace
                BEFORE INSERT ON telemetry_events
                WHEN EXISTS(SELECT 1 FROM telemetry_events
                    WHERE plant_id=NEW.plant_id AND device_id=NEW.device_id
                    AND event_id=NEW.event_id)
                BEGIN SELECT RAISE(ABORT, 'Telemetry event is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS telemetry_seeds_no_replace
                BEFORE INSERT ON telemetry_seeds
                WHEN EXISTS(SELECT 1 FROM telemetry_seeds WHERE plant_id=NEW.plant_id)
                BEGIN SELECT RAISE(ABORT, 'Telemetry seed already exists'); END;
            """)
            # Identifiers below are fixed implementation constants, never model input.
            for table in ("telemetry_seeds", "telemetry_events", "telemetry_samples"):
                for operation in ("UPDATE", "DELETE"):
                    db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                        BEFORE {operation} ON {table}
                        BEGIN SELECT RAISE(ABORT, 'Telemetry is immutable'); END""")

    @staticmethod
    def _write_event(db: sqlite3.Connection, metadata: dict, signals: dict) -> None:
        """Internal, transactional fixture writer; insert the seal after all samples."""
        from .schemas import canonical

        count = sum(len(points) for points in signals.values())
        if count > MAX_EVENT_POINTS:
            raise DomainError("telemetry_limit", "Event exceeds the 10000-point bound")
        key = (metadata["plant_id"], metadata["device_id"], metadata["event_id"])
        for signal, points in signals.items():
            for ordinal, point in enumerate(points):
                db.execute("INSERT INTO telemetry_samples VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    *key, signal, ordinal, point["timestamp_ms"], canonical(point["value"]),
                    point["unit"], point["quality"], int(point["sensor_valid"]),
                ))
        db.execute("INSERT INTO telemetry_events VALUES(?,?,?,?,?,?)", (
            *key, canonical(metadata), digest({**metadata, "signals": signals}), count,
        ))

    def seed(self, plant_id: str = "PLANT-A") -> dict:
        """Create six fixtures plus device/plant isolation fixtures, atomically, once.

        Case names exist only in this seed listing, not in stored event metadata or
        snapshots. They are demo/test fixture labels, never verification outcomes.
        """
        from .schemas import canonical

        if not isinstance(plant_id, str) or not _IDENTIFIER.fullmatch(plant_id):
            raise DomainError("invalid_plant", "Invalid plant identifier")
        parameters = {
            "jam": (1.5, 0.06, 0.5, 1.0),
            "slip": (0.5, 1.5, 0.3, 1.0),
            "sensor_fault": (0.25, 1.0, 1.0, 2.0),
            "normal": (0.25, 1.0, 1.0, 1.0),
            "missing": (0.25, 1.0, 1.0, 1.0),
            "conflicting": (1.5, 1.5, 0.025, 1.0),
        }
        other_plant = "PLANT-B" if plant_id != "PLANT-B" else "PLANT-A"
        listing: dict[str, Any] = {
            "synthetic": True, "plant_id": plant_id, "cases": {}, "isolation_cases": {},
        }
        fixtures = [
            (name, plant_id, "CONVEYOR-01", f"EVT-{index:03d}", values)
            for index, (name, values) in enumerate(parameters.items(), start=1)
        ]
        fixtures += [
            ("other_device", plant_id, "CONVEYOR-02", "EVT-007", parameters["normal"]),
            ("other_plant", other_plant, "CONVEYOR-01", "EVT-008", parameters["slip"]),
        ]
        with self.store.transaction() as db:
            if db.execute("SELECT 1 FROM telemetry_seeds WHERE plant_id=?", (plant_id,)).fetchone():
                raise DomainError("telemetry_seed_exists", "Synthetic seed already exists", 409)
            try:
                for name, plant, device, event, values in fixtures:
                    metadata = _metadata(plant, device, event)
                    signals = _generate(**dict(zip(
                        ("load", "shaft_speed", "transfer", "encoder_gain"), values, strict=True,
                    )))
                    if name == "missing":
                        signals["motor_current"] = [
                            p for p in signals["motor_current"]
                            if not 10_000 <= p["timestamp_ms"] < 15_000
                        ]
                        signals["encoder_speed"] = []
                    self._write_event(db, metadata, signals)
                    group = "cases" if name in parameters else "isolation_cases"
                    listing[group][name] = {
                        "plant_id": plant, "device_id": device, "event_id": event,
                        "window": dict(metadata["event_window"]),
                        "title": metadata["title"], "complaint": metadata["complaint"],
                    }
                db.execute("INSERT INTO telemetry_seeds VALUES(?,?)", (plant_id, canonical(listing)))
            except sqlite3.IntegrityError as exc:
                raise DomainError("telemetry_seed_exists", "Seed conflicts with immutable data", 409) from exc
        return listing

    def demo_cases(self, actor: Actor) -> list[dict]:
        """List existing authorized synthetic fixtures for the UI, never reseeding.

        case_label is a demo-only fixture label. Callers must omit it from diagnosis
        requests and model context, just as they omit the seed listing itself.
        """
        cases = []
        fields = ("plant_id", "device_id", "event_id", "window", "title", "complaint")
        with self.store.connect() as db:
            current = self.store.current_actor(actor, db=db)
            rows = db.execute("SELECT listing_json FROM telemetry_seeds ORDER BY plant_id")
            try:
                for row in rows:
                    listing = json.loads(row["listing_json"])
                    if listing["synthetic"] is not True:
                        raise ValueError("Seed lacks synthetic provenance")
                    for group in ("cases", "isolation_cases"):
                        for label, metadata in listing[group].items():
                            # Isolation fixtures can belong to a different plant
                            # than the enclosing seed, so scope each case itself.
                            if metadata["plant_id"] not in current.plants:
                                continue
                            cases.append({
                                "case_label": label,
                                **{field: metadata[field] for field in fields},
                                "synthetic": True,
                            })
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise DomainError("telemetry_integrity", "Malformed synthetic seed listing", 409) from exc
            self.store.current_actor(actor, db=db)
        return sorted(cases, key=lambda case: (
            case["plant_id"], case["device_id"], case["event_id"], case["case_label"],
        ))

    def event(
        self, plant_id: str, device_id: str, event_id: str, window: Window, actor: Actor,
    ) -> dict:
        """Read a scoped [start,end) snapshot; never include points outside the window."""
        window = _window(window)
        key = (plant_id, device_id, event_id)
        with self.store.connect() as db:
            self.store.authorize(actor, plant_id, db=db)
            row = db.execute(
                "SELECT * FROM telemetry_events WHERE plant_id=? AND device_id=? AND event_id=?",
                key,
            ).fetchone()
            if row is None:
                raise DomainError("telemetry_not_found", "No event in the requested scope", 404)
            try:
                metadata = json.loads(row["metadata_json"])
                event_window = _window(metadata["event_window"])
                if tuple(metadata[k] for k in ("plant_id", "device_id", "event_id")) != key:
                    raise ValueError("Event identity mismatch")
                if not event_window.start_ms <= window.start_ms < window.end_ms <= event_window.end_ms:
                    raise DomainError("telemetry_window_mismatch", "Window is outside the event interval")
                count = db.execute(
                    "SELECT COUNT(*) FROM telemetry_samples "
                    "WHERE plant_id=? AND device_id=? AND event_id=?", key,
                ).fetchone()[0]
                if count > MAX_EVENT_POINTS or count != row["sample_count"]:
                    raise DomainError("telemetry_integrity", "Event sample count changed", 409)
                # Preserve acquisition order, so timestamp disorder cannot be hidden by sorting.
                samples = db.execute(
                    "SELECT signal,timestamp_ms,value_json,unit,quality,sensor_valid "
                    "FROM telemetry_samples WHERE plant_id=? AND device_id=? AND event_id=? "
                    "ORDER BY signal,ordinal LIMIT ?", (*key, MAX_EVENT_POINTS),
                ).fetchall()
                signals: dict[str, list[dict]] = {name: [] for name in metadata["units"]}
                for sample in samples:
                    signals.setdefault(sample["signal"], []).append({
                        "timestamp_ms": sample["timestamp_ms"],
                        "value": json.loads(sample["value_json"]),
                        "unit": sample["unit"], "quality": sample["quality"],
                        "sensor_valid": (
                            bool(sample["sensor_valid"])
                            if sample["sensor_valid"] in (0, 1) else sample["sensor_valid"]
                        ),
                    })
                if digest({**metadata, "signals": signals}) != row["data_hash"]:
                    raise DomainError("telemetry_integrity", "Immutable event content hash mismatch", 409)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise DomainError("telemetry_integrity", "Malformed immutable event", 409) from exc
        snapshot = {
            **metadata,
            "window": window.model_dump(),
            "event_data_hash": row["data_hash"],
            "signals": {
                name: [p for p in points if window.start_ms <= p["timestamp_ms"] < window.end_ms]
                for name, points in signals.items()
            },
        }
        snapshot["data_hash"] = digest(snapshot)
        return snapshot

    @staticmethod
    def verify(snapshot: dict, condition: Condition) -> dict:
        return verify(snapshot, condition)


def verify(snapshot: dict, condition: Condition) -> dict:
    """Compute signal evidence, never a diagnosis, from canonical snapshot samples.

    difference_mean is signed mean(signal - other_signal) with exact timestamp
    pairing. gt/lt are strict; between includes both endpoints; eq is exact.
    Freshness is evaluated at the requested historical window end, not wall time.
    """
    if not isinstance(condition, Condition):
        raise DomainError("invalid_condition", "A validated Condition is required")
    try:
        condition = Condition.model_validate(condition.model_dump())
    except ValidationError as exc:
        raise DomainError("invalid_condition", "Invalid signal condition") from exc
    if condition.source != "signal":
        raise DomainError("signal_condition_required", "Human observations are not telemetry signals")
    details: dict[str, Any] = {
        "computed": None, "n": 0, "signal": condition.signal,
        "other_signal": condition.other_signal, "statistic": condition.statistic,
        "operator": condition.operator, "threshold": condition.threshold,
        "upper": condition.upper, "unit": condition.unit,
    }

    def missing(reason: str, **extra: Any) -> dict:
        return {"state": "missing", "details": {**details, "reason": reason, **extra}}

    if not isinstance(snapshot, dict):
        return missing("invalid_snapshot")
    signals = snapshot.get("signals")
    if not isinstance(signals, dict) or any(not isinstance(v, list) for v in signals.values()):
        return missing("invalid_signals")
    if len(signals) > len(SIGNAL_UNITS) or sum(map(len, signals.values())) > MAX_EVENT_POINTS:
        return missing("point_limit")
    try:
        payload = {key: value for key, value in snapshot.items() if key != "data_hash"}
        if not isinstance(snapshot.get("data_hash"), str) or digest(payload) != snapshot["data_hash"]:
            return missing("snapshot_hash_mismatch")
        if (snapshot.get("schema_version") != _SCHEMA or snapshot.get("synthetic") is not True
                or snapshot.get("provenance") != _PROVENANCE):
            return missing("invalid_provenance")
        if any(not isinstance(snapshot.get(k), str) or not _IDENTIFIER.fullmatch(snapshot[k])
               for k in ("plant_id", "device_id", "event_id")):
            return missing("invalid_identity")
        window = _window(snapshot["window"])
        event_window = _window(snapshot["event_window"])
        if not event_window.start_ms <= window.start_ms < window.end_ms <= event_window.end_ms:
            return missing("window_mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot.get("event_data_hash", "")):
            return missing("invalid_event_hash")
    except (DomainError, KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return missing("invalid_snapshot")
    details["data_hash"] = snapshot["data_hash"]
    details["window"] = window.model_dump()
    unit = SIGNAL_UNITS[condition.signal]
    units = snapshot.get("units")
    required = [condition.signal]
    if condition.other_signal is not None:
        required.append(condition.other_signal)
    if (condition.unit != unit or not isinstance(units, dict)
            or any(SIGNAL_UNITS[name] != unit or units.get(name) != unit for name in required)):
        return missing("unit_mismatch")
    series: list[list[dict]] = []
    for name in required:
        points = signals.get(name)
        if not points:
            return missing("signal_absent", invalid_signal=name)
        details["n"] = len(points)
        previous = None
        for point in points:
            if not isinstance(point, dict):
                return missing("invalid_sample", invalid_signal=name)
            timestamp, value = point.get("timestamp_ms"), point.get("value")
            if type(timestamp) is not int or not 0 <= timestamp <= 2**53 - 1:
                return missing("invalid_timestamp", invalid_signal=name)
            if not window.start_ms <= timestamp < window.end_ms:
                return missing("sample_outside_window", invalid_signal=name)
            if previous is not None and timestamp <= previous:
                return missing("nonmonotonic_timestamp", invalid_signal=name)
            previous = timestamp
            if point.get("unit") != unit:
                return missing("unit_mismatch", invalid_signal=name)
            if point.get("quality") != "good" or point.get("sensor_valid") is not True:
                return missing("invalid_sensor", invalid_signal=name)
            if type(value) not in (int, float):
                return missing("invalid_value", invalid_signal=name)
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or (unit == "bool" and value not in (0, 1)):
                return missing("invalid_value", invalid_signal=name)
        if len(points) < condition.min_samples:
            return missing("insufficient_samples", invalid_signal=name)
        gaps = [points[0]["timestamp_ms"] - window.start_ms,
                window.end_ms - points[-1]["timestamp_ms"]]
        gaps.extend(b["timestamp_ms"] - a["timestamp_ms"]
                    for a, b in zip(points, points[1:], strict=False))
        max_gap = max(gaps)
        details["max_gap_ms"] = max_gap
        if max_gap > condition.max_gap_ms:
            return missing("coverage_gap", invalid_signal=name)
        age = window.end_ms - points[-1]["timestamp_ms"]
        details["latest_timestamp_ms"] = points[-1]["timestamp_ms"]
        details["age_ms"] = age
        if age > condition.max_age_ms:
            return missing("stale_samples", invalid_signal=name)
        series.append(points)
    try:
        values = [p["value"] for p in series[0]]
        if condition.statistic == "difference_mean":
            if [p["timestamp_ms"] for p in series[0]] != [p["timestamp_ms"] for p in series[1]]:
                return missing("unpaired_timestamps")
            values = [a["value"] - b["value"] for a, b in zip(*series, strict=True)]
        if condition.statistic in ("mean", "difference_mean"):
            computed = math.fsum(values) / len(values)
        elif condition.statistic == "max":
            computed = max(values)
        else:
            computed = min(values)
        if not math.isfinite(computed):
            return missing("nonfinite_computation")
    except (OverflowError, ValueError):
        return missing("nonfinite_computation")
    details["computed"] = computed
    details["n"] = len(values)
    if condition.operator == "gt":
        supported = computed > condition.threshold
    elif condition.operator == "lt":
        supported = computed < condition.threshold
    elif condition.operator == "between":
        supported = condition.threshold <= computed <= condition.upper
    else:
        supported = computed == condition.threshold
    return {"state": "supported" if supported else "refuted", "details": details}
