"""Original, offline report-to-OWL knowledge builds and explicit human publication.

The ontology checks structural compatibility of decision rules, not physical causality.
Neither a model draft nor successful classification authorizes publication. Each HermiT
invocation runs in a disposable worker; Owlready2's Java globals never change in the server.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import ValidationError

from .schemas import (
    Actor,
    DomainError,
    KnowledgeBuildRequest,
    KnowledgeDraft,
    PublishRequest,
    ReportSource,
    canonical,
    digest,
)
from .storage import Store, new_id, now

ONTOLOGY_IRI = "urn:original:conveyor"
OWL = "http://www.w3.org/2002/07/owl#"
MAX_DOCUMENT_BYTES = 256_000
MAX_OUTPUT_BYTES = 65_536
MAX_OWL_BYTES = 2_000_000
JAVA_HEAP_MB = 256


class LocalDraftExtractor(Protocol):
    """An explicitly local, untrusted draft adapter; it has no Store or publish capability.

    Implementations must run only a local model. This declaration is not an OS sandbox;
    configure trusted adapter code, and treat everything the model returns as untrusted.
    """

    local_only: bool

    def extract(self, reports: list[ReportSource]) -> dict[str, Any] | str: ...


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML mappings require unique string keys")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _safe_yaml(text: str):
    if not isinstance(text, str) or len(text.encode()) > MAX_DOCUMENT_BYTES:
        raise DomainError("invalid_draft", "Draft document exceeds the local size budget")
    try:
        depth = 0
        for count, event in enumerate(yaml.parse(text)):
            if count > 20_000 or isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                raise ValueError("YAML aliases, anchors, or oversized documents are not allowed")
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > 32:
                    raise ValueError("YAML nesting exceeds the local budget")
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
        return yaml.load(text, Loader=_UniqueSafeLoader)
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise DomainError("invalid_draft", "Invalid safe YAML draft") from exc


def _validated_reports(reports: list[ReportSource]) -> list[ReportSource]:
    try:
        values = [ReportSource.model_validate(r.model_dump() if isinstance(r, ReportSource) else r)
                  for r in reports]
    except (ValidationError, TypeError, AttributeError) as exc:
        raise DomainError("invalid_draft", "Invalid original report sources") from exc
    if not 1 <= len(values) <= 16 or len({r.report_id for r in values}) != len(values):
        raise DomainError("invalid_draft", "Provide between one and sixteen unique reports")
    return values


def draft_from_yaml(text: str, reports: list[ReportSource]) -> KnowledgeDraft:
    """Safe-parse a full structured draft and bind it to its original source reports."""
    return _validate_extracted(_safe_yaml(text), _validated_reports(reports))


def _validate_extracted(value, reports):
    try:
        draft = KnowledgeDraft.model_validate(value)
        KnowledgeBuildRequest(request_id="draft-validation", draft=draft, reports=reports)
        return draft
    except (ValidationError, ValueError, TypeError) as exc:
        raise DomainError("invalid_draft", "Draft schema or source provenance does not match") from exc


def draft_from_reports(
    reports: list[ReportSource], *, extractor: LocalDraftExtractor | None = None
) -> KnowledgeDraft:
    """Extract structured knowledge fragments from original YAML reports, or a local model.

    Native reports contain a prose ``summary`` and a ``knowledge`` mapping. Fragments
    supply nodes/conditions and agreeing metadata; report IDs are derived, never invented.
    A local model may instead return a full draft mapping or safe YAML. Both paths stop at
    Pydantic validation and never build or publish knowledge implicitly.
    """
    reports = _validated_reports(reports)
    if extractor is not None:
        if getattr(extractor, "local_only", False) is not True:
            raise DomainError("nonlocal_extractor", "Only an explicitly local draft adapter is allowed")
        try:
            value = extractor.extract([r.model_copy(deep=True) for r in reports])
        except Exception as exc:
            raise DomainError("draft_extraction_failed", "Local draft extraction failed") from exc
        if isinstance(value, str):
            value = _safe_yaml(value)
        try:
            if len(canonical(value).encode()) > MAX_DOCUMENT_BYTES:
                raise ValueError("Oversized extraction")
        except (ValueError, TypeError, RecursionError) as exc:
            raise DomainError("invalid_draft", "Local extraction exceeds the draft budget") from exc
        return _validate_extracted(value, reports)

    combined: dict[str, Any] = {"nodes": [], "conditions": [], "global_required": []}
    allowed = {"tree_id", "plant_id", "title", "root_id", "nodes", "conditions", "global_required"}
    for report in reports:
        document = _safe_yaml(report.text)
        if (not isinstance(document, dict) or set(document) != {"summary", "knowledge"}
                or not isinstance(document["summary"], str)
                or not isinstance(document["knowledge"], dict)):
            raise DomainError("invalid_draft", "Report requires summary and knowledge fragments")
        fragment = document["knowledge"]
        if not set(fragment) <= allowed:
            raise DomainError("invalid_draft", "Unknown report knowledge field")
        for key, value in fragment.items():
            if key in {"nodes", "conditions", "global_required"}:
                if not isinstance(value, list):
                    raise DomainError("invalid_draft", "Fragment collection must be a list")
                combined[key].extend(value)
            elif key in combined and combined[key] != value:
                raise DomainError("invalid_draft", "Reports disagree on knowledge metadata")
            else:
                combined[key] = value
    combined["source_report_ids"] = [r.report_id for r in reports]
    return _validate_extracted(combined, reports)


def request_from_reports(
    request_id: str, reports: list[ReportSource], *, extractor: LocalDraftExtractor | None = None
) -> KnowledgeBuildRequest:
    """Return an untrusted build request, with no publication side effect."""
    reports = _validated_reports(reports)
    return KnowledgeBuildRequest(request_id=request_id,
                                 draft=draft_from_reports(reports, extractor=extractor), reports=reports)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(128_000), b""):
            value.update(chunk)
    return value.hexdigest()


def _check_local_owl(text: str) -> None:
    """Reject imports and external XML entities before any ontology loader is called."""
    if not isinstance(text, str) or len(text.encode()) > MAX_OWL_BYTES:
        raise DomainError("invalid_ontology", "OWL document exceeds the local budget")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise DomainError("invalid_ontology", "External entities and document types are forbidden")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise DomainError("invalid_ontology", "Malformed OWL serialization") from exc
    if any(item.tag == "{" + OWL + "}imports" for item in root.iter()):
        raise DomainError("ontology_import_forbidden", "OWL builds cannot import any ontology")


def export_owl(draft: KnowledgeDraft) -> str:
    """Export an original structural ontology. Export alone is never a reasoner check."""
    import io

    import owlready2 as owl

    draft = KnowledgeDraft.model_validate(draft.model_dump())
    world = owl.World()
    ontology = world.get_ontology(ONTOLOGY_IRI + "#")
    try:
        with ontology:
            def cls(name, *bases):
                return types.new_class(name, bases or (owl.Thing,))

            tree_node = cls("TreeNode")
            branch = cls("BranchNode", tree_node)
            leaf = cls("LeafNode", tree_node)
            diagnosis = cls("DiagnosisNode", leaf)
            normal = cls("NormalNode", leaf)
            condition = cls("Condition")
            numeric = cls("SignalCondition", condition)
            human = cls("HumanCondition", condition)
            gate = cls("RuleGate")
            all_gate = cls("AllGate", gate)
            any_gate = cls("AnyGate", gate)
            source = cls("OriginalReport")
            owl.AllDisjoint([tree_node, condition, gate, source])
            owl.AllDisjoint([branch, leaf])
            owl.AllDisjoint([diagnosis, normal])
            owl.AllDisjoint([numeric, human])
            owl.AllDisjoint([all_gate, any_gate])

            def obj(name, domain, range_, functional=False):
                bases = (owl.ObjectProperty, owl.FunctionalProperty) if functional else (owl.ObjectProperty,)
                prop = types.new_class(name, bases)
                prop.domain, prop.range = [domain], [range_]
                return prop

            child = obj("hasChild", tree_node, tree_node)
            parent = obj("hasParent", tree_node, tree_node, True)
            child.inverse_property = parent
            predicate = obj("hasCondition", leaf, condition)
            mandatory = obj("requiresObservation", leaf, condition)
            global_required = obj("globallyRequires", tree_node, condition)
            selected_gate = obj("usesGate", leaf, gate, True)
            provenance = obj("sourcedFrom", tree_node, source)
            branch.is_a.append(child.min(1, tree_node))
            leaf.is_a.append(child.max(0, tree_node))

            evidence_leaf = cls("EvidenceBoundLeaf", leaf)
            evidence_leaf.equivalent_to = [leaf & predicate.some(condition)]
            numeric_leaf = cls("NumericEvidenceLeaf", leaf)
            numeric_leaf.equivalent_to = [leaf & predicate.some(numeric)]
            guarded = cls("HumanGuardedDiagnosis", diagnosis)
            guarded.equivalent_to = [diagnosis & mandatory.some(human)]

            def data(name, domain, range_):
                prop = types.new_class(name, (owl.DataProperty, owl.FunctionalProperty))
                prop.domain, prop.range = [domain], [range_]
                return prop

            fields = {name: data(name, condition, str) for name in
                      ("sourceKind", "signalName", "statisticName", "operatorName", "unitName", "humanKey")}
            threshold = data("numericThreshold", numeric, float)
            upper = data("numericUpper", numeric, float)
            human_value = data("observationValue", human, bool)
            samples = data("minimumSamples", condition, int)
            gap = data("maximumGapMs", condition, int)
            age = data("maximumAgeMs", condition, int)
            other = data("otherSignalName", numeric, str)
            gates = {"all": all_gate("gate_all"), "any": any_gate("gate_any")}
            conditions = {}
            for entry in draft.conditions:
                individual = (numeric if entry.source == "signal" else human)("condition_" + entry.condition_id)
                individual.label = [entry.label]
                conditions[entry.condition_id] = individual
                for key, val in (("sourceKind", entry.source), ("signalName", entry.signal),
                                 ("statisticName", entry.statistic), ("operatorName", entry.operator),
                                 ("unitName", entry.unit), ("humanKey", entry.human_key)):
                    if val is not None:
                        fields[key][individual] = [val]
                if entry.source == "signal":
                    threshold[individual] = [float(entry.threshold)]
                    if entry.upper is not None:
                        upper[individual] = [entry.upper]
                    if entry.other_signal is not None:
                        other[individual] = [entry.other_signal]
                else:
                    human_value[individual] = [entry.threshold]
                samples[individual], gap[individual], age[individual] = (
                    [entry.min_samples], [entry.max_gap_ms], [entry.max_age_ms])
            reports = [source("report_" + rid) for rid in draft.source_report_ids]
            nodes, node_classes = {}, {}
            for node in draft.nodes:
                base = {"branch": branch, "diagnosis": diagnosis, "normal": normal}[node.kind]
                node_cls = cls("rule_" + node.node_id, base)
                node_classes[node.node_id] = node_cls
                nodes[node.node_id] = node_cls("node_" + node.node_id)
                nodes[node.node_id].label = [node.label]
            # Distinct IDs have distinct individuals; OWL's open-world semantics alone do not.
            owl.AllDifferent(list(nodes.values()) + list(conditions.values()) + reports + list(gates.values()))
            for node in draft.nodes:
                individual, node_cls = nodes[node.node_id], node_classes[node.node_id]
                provenance[individual] = reports
                global_required[individual] = [conditions[c] for c in draft.global_required]
                if node.kind == "branch":
                    children = [nodes[c] for c in node.children]
                    child[individual] = children
                    node_cls.is_a.extend([child.exactly(len(children), tree_node), child.only(owl.OneOf(children))])
                else:
                    predicate[individual] = [conditions[c] for c in node.conditions]
                    mandatory[individual] = [conditions[c] for c in node.mandatory]
                    selected_gate[individual] = [gates[node.gate]]
                    node_cls.is_a.extend(predicate.value(conditions[c]) for c in node.conditions)
                    node_cls.is_a.extend(mandatory.value(conditions[c]) for c in node.mandatory)
                    node_cls.is_a.append(selected_gate.value(gates[node.gate]))
            ontology.metadata.comment.append(
                "Original synthetic rule structure. All/any gates and numeric thresholds describe "
                "evidence requirements, not established physical causality. Human review is required.")
        stream = io.BytesIO()
        ontology.save(file=stream, format="rdfxml")
        text = stream.getvalue().decode()
        _check_local_owl(text)
        return text
    finally:
        world.close()


class _BudgetExceeded(Exception):
    def __init__(self, code: str):
        self.code = code


def _run_bounded(command, *, timeout, output_budget=MAX_OUTPUT_BYTES, env=None, group=False):
    """Bound wall time and combined pipe output; kill our entire worker group on failure."""
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, env=env, start_new_session=group)
    output = bytearray()
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise _BudgetExceeded("timeout")
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif len(output) + len(chunk) > output_budget:
                    raise _BudgetExceeded("output_limit")
                else:
                    output.extend(chunk)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise _BudgetExceeded("timeout")
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise _BudgetExceeded("timeout") from exc
        return code, bytes(output)
    finally:
        selector.close()
        if group:
            # Also kill a descendant retaining a pipe after its worker exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def _reasoner_base(java_path: str | None, timeout_seconds: float) -> dict:
    return {"name": "HermiT", "engine": "owlready2.sync_reasoner_hermit", "status": "not_run",
            "success": False, "consistent": None, "unsatisfiable_classes": [],
            "java_path": java_path, "timeout_seconds": timeout_seconds,
            "java_heap_mb": JAVA_HEAP_MB, "output_budget_bytes": MAX_OUTPUT_BYTES,
            "ontology_iri": ONTOLOGY_IRI, "inferences": [], "reload_verified": False,
            "scope": "Structural rule consistency and classification, not physical causal truth"}


def _resolve_java(java_path):
    supplied = java_path or os.environ.get("INDUSTRIAL_DIAGNOSIS_JAVA") or os.environ.get("JAVA_PATH")
    if not supplied and os.environ.get("JAVA_HOME"):
        supplied = str(Path(os.environ["JAVA_HOME"]) / "bin" / "java")
    if not supplied:
        return None
    path = Path(supplied).expanduser()
    if not path.is_absolute():
        raise DomainError("invalid_java_config", "Configure an explicit absolute Java binary path")
    return str(path.resolve())


def _classify(text: str, draft: KnowledgeDraft, java_path: str | None, timeout_seconds: float):
    report = _reasoner_base(java_path, timeout_seconds)
    report["input_owl_sha256"] = _sha(text.encode())
    _check_local_owl(text)
    if java_path is None or not Path(java_path).is_file() or not os.access(java_path, os.X_OK):
        report.update(status="missing_jvm", error="An explicit executable JVM is required")
        return report, None
    expected = []
    conditions = {condition.condition_id: condition for condition in draft.conditions}
    for node in draft.nodes:
        if node.kind == "branch":
            continue
        superclasses = ["EvidenceBoundLeaf"]
        if any(conditions[c].source == "signal" for c in node.conditions):
            superclasses.append("NumericEvidenceLeaf")
        if node.kind == "diagnosis" and any(conditions[c].source == "human" for c in node.mandatory):
            superclasses.append("HumanGuardedDiagnosis")
        expected.extend({"subclass": ONTOLOGY_IRI + "#rule_" + node.node_id,
                         "superclass": ONTOLOGY_IRI + "#" + superclass}
                        for superclass in superclasses)
    with tempfile.TemporaryDirectory(prefix="original-hermit-") as temporary:
        directory = Path(temporary)
        input_path, output_path = directory / "input.owl", directory / "classified.owl"
        result_path, config_path = directory / "result.json", directory / "config.json"
        input_path.write_text(text)
        config = {"java_path": java_path, "timeout_seconds": timeout_seconds,
                  "expected": expected, "input_owl_sha256": report["input_owl_sha256"]}
        config_path.write_text(canonical(config))
        command = [sys.executable, "-m", "industrial_diagnosis.knowledge", "--hermit-worker",
                   str(input_path), str(output_path), str(result_path), str(config_path)]
        # Do not inherit model/API credentials, JVM option injections, or user Python paths.
        env = {"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
               "LANG": "C.UTF-8", "TMPDIR": temporary, "PYTHONHASHSEED": "0"}
        try:
            code, log = _run_bounded(command, timeout=timeout_seconds, env=env, group=True)
            report["worker_output_sha256"] = _sha(log)
            if code != 0 or not result_path.is_file() or result_path.stat().st_size > MAX_OUTPUT_BYTES:
                report.update(status="reasoner_error", error="HermiT worker did not return a valid result",
                              worker_exit_code=code, worker_output=log.decode(errors="replace")[:2000])
                return report, None
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict):
                raise ValueError("Invalid worker report")
            report.update(result)
            if report["status"] != "success":
                report["success"] = False
                return report, None
            if (not report.get("success") or report.get("consistent") is not True
                    or report.get("unsatisfiable_classes") or not report.get("reload_verified")
                    or not report.get("inferences") or report.get("hermit_exit_code") != 0
                    or report.get("input_owl_sha256") != config["input_owl_sha256"]
                    or not report.get("hermit_jar_sha256") or not report.get("java_binary_sha256")):
                raise ValueError("Incomplete reasoner attestation")
            if not output_path.is_file() or output_path.stat().st_size > MAX_OWL_BYTES:
                raise ValueError("Missing classified ontology")
            classified = output_path.read_text()
            _check_local_owl(classified)
            if _sha(classified.encode()) != report.get("classified_owl_sha256"):
                raise ValueError("Classified ontology hash mismatch")
            return report, classified
        except _BudgetExceeded as exc:
            report.update(status=exc.code, success=False, error="HermiT exceeded its finite worker budget")
        except (OSError, ValueError, DomainError) as exc:
            report.update(status="reasoner_error", success=False, error=str(exc)[:1000])
        return report, None


def _hermit_worker(input_path: Path, output_path: Path, result_path: Path, config_path: Path):
    """Private subprocess entry point. Only this disposable process mutates Java globals."""
    import resource

    import owlready2 as owl
    import owlready2.reasoning as reasoning

    config = json.loads(config_path.read_text())
    timeout = float(config["timeout_seconds"])
    deadline = time.monotonic() + timeout
    resource.setrlimit(resource.RLIMIT_FSIZE, (4_000_000, 4_000_000))
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout) + 2, math.ceil(timeout) + 3))
    # Java additionally has explicit heap, metaspace, code cache and stack limits below.
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    java_path = config["java_path"]
    report = _reasoner_base(java_path, timeout)
    report["input_owl_sha256"] = config["input_owl_sha256"]
    world = None
    try:
        text = input_path.read_text()
        _check_local_owl(text)
        if _sha(text.encode()) != config["input_owl_sha256"]:
            raise ValueError("Input ontology hash mismatch")
        report["java_binary_sha256"] = _file_sha(Path(java_path))
        code, version = _run_bounded([java_path, "-version"], timeout=min(5, deadline - time.monotonic()))
        report["java_version"] = version.decode(errors="replace")[:2000]
        if code != 0 or b"version" not in version.lower() or not any(
            name in version.lower() for name in (b"java", b"openjdk")
        ):
            report.update(status="missing_jvm", error="Configured binary did not start a Java runtime")
            return
        jar = Path(owl.__file__).parent / "hermit" / "HermiT.jar"
        report.update(owlready2_version=getattr(owl, "VERSION", "unknown"),
                      hermit_jar_sha256=_file_sha(jar))
        with zipfile.ZipFile(jar) as archive:
            manifest = archive.read("META-INF/MANIFEST.MF").decode(errors="replace")
        report["hermit_manifest"] = manifest[:3000]
        owl.JAVA_EXE = java_path
        reasoning.JAVA_MEMORY = JAVA_HEAP_MB

        def bounded_hermit(command, **kwargs):
            if command[0] != java_path or "org.semanticweb.HermiT.cli.CommandLine" not in command:
                raise ValueError("Unexpected reasoner command")
            command[1:1] = ["-XX:MaxMetaspaceSize=128m", "-XX:ReservedCodeCacheSize=64m",
                            "-XX:CompressedClassSpaceSize=64m", "-Xss1m", "-XX:ActiveProcessorCount=2"]
            report["hermit_command"] = command
            code, output = _run_bounded(command, timeout=deadline - time.monotonic())
            report.update(hermit_exit_code=code, hermit_output_sha256=_sha(output),
                          hermit_output_bytes=len(output), hermit_output=output.decode(errors="replace")[:2000])
            if code:
                raise subprocess.CalledProcessError(code, command, output=output)
            return output

        # Replace only this worker module's subprocess facade, not the subprocess module.
        reasoning.subprocess = types.SimpleNamespace(check_output=bounded_hermit, STDOUT=subprocess.STDOUT,
                                                       CalledProcessError=subprocess.CalledProcessError)
        world = owl.World()
        with input_path.open("rb") as stream:
            ontology = world.get_ontology(ONTOLOGY_IRI + "#").load(fileobj=stream)
        expected = config["expected"]
        if not expected:
            raise ValueError("At least one meaningful inference is required")
        for pair in expected:
            sub, sup = world[pair["subclass"]], world[pair["superclass"]]
            if sub is None or sup is None or sup in sub.ancestors():
                raise ValueError("Required inference is missing or already asserted")
        # This call executes the bundled HermiT JVM classifier, not a syntax-only substitute.
        owl.sync_reasoner_hermit(ontology, debug=0)
        unsatisfiable = sorted(c.iri for c in world.inconsistent_classes() if c != owl.Nothing)
        report["unsatisfiable_classes"] = unsatisfiable
        if unsatisfiable:
            report.update(status="unsatisfiable", consistent=True, error="Named classes are unsatisfiable")
            return
        for pair in expected:
            if world[pair["superclass"]] not in world[pair["subclass"]].ancestors():
                raise ValueError("HermiT did not derive a required rule classification")
        report["consistent"] = True
        report["inferences"] = expected
        ontology.save(file=str(output_path), format="rdfxml")
        classified = output_path.read_text()
        _check_local_owl(classified)
        report["classified_owl_sha256"] = _sha(classified.encode())
        # Close the classified world and reload actual bytes in a genuinely fresh world.
        world.close()
        world = owl.World()
        with output_path.open("rb") as stream:
            world.get_ontology(ONTOLOGY_IRI + "#").load(fileobj=stream)
        for pair in expected:
            if world[pair["superclass"]] not in world[pair["subclass"]].ancestors():
                raise ValueError("Saved classification did not survive ontology reload")
        if any(c != owl.Nothing for c in world.inconsistent_classes()):
            raise ValueError("Reload contains an unsatisfiable named class")
        report.update(status="success", success=True, reload_verified=True)
    except owl.OwlReadyInconsistentOntologyError:
        report.update(status="inconsistent", consistent=False, error="HermiT found an inconsistent ontology")
    except _BudgetExceeded as exc:
        report.update(status=exc.code, error="HermiT exceeded its finite execution budget")
    except FileNotFoundError:
        report.update(status="missing_jvm", error="Configured JVM or packaged reasoner is unavailable")
    except Exception as exc:
        report.update(status="reasoner_error", error=str(exc)[:1500])
    finally:
        if world is not None:
            world.close()
        result_path.write_text(canonical(report))


class KnowledgeStore:
    def __init__(self, store: Store, java_path=None, timeout_seconds=30):
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not (
            math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 300
        ):
            raise DomainError("invalid_reasoner_config", "Reasoner timeout must be finite, positive and at most 300s")
        self.store = store
        self.java_path = _resolve_java(java_path)
        self.timeout_seconds = float(timeout_seconds)
        with store.transaction() as db:
            # Separate immutable build/publication records from the mutable active pointer.
            for statement in (
                """CREATE TABLE IF NOT EXISTS knowledge_versions(
                    version_id TEXT PRIMARY KEY, plant_id TEXT NOT NULL, owner TEXT NOT NULL,
                    request_id TEXT NOT NULL, request_hash TEXT NOT NULL, content_hash TEXT NOT NULL,
                    build_status TEXT NOT NULL CHECK(build_status IN ('checked','blocked')),
                    manifest_json TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(owner,request_id))""",
                """CREATE TABLE IF NOT EXISTS knowledge_publications(
                    owner TEXT NOT NULL, action_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                    version_id TEXT NOT NULL REFERENCES knowledge_versions(version_id),
                    review_note TEXT NOT NULL, result_json TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(owner,action_id))""",
                """CREATE TABLE IF NOT EXISTS knowledge_active(
                    plant_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES knowledge_versions(version_id),
                    revision INTEGER NOT NULL, activated_at REAL NOT NULL)""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_active_checked_insert BEFORE INSERT ON knowledge_active
                    WHEN NOT EXISTS(SELECT 1 FROM knowledge_versions v JOIN knowledge_publications p
                        ON p.version_id=v.version_id WHERE v.version_id=NEW.version_id
                        AND v.plant_id=NEW.plant_id AND v.build_status='checked')
                    BEGIN SELECT RAISE(ABORT,'Active knowledge must be a checked scoped publication'); END""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_active_checked_update BEFORE UPDATE ON knowledge_active
                    WHEN NOT EXISTS(SELECT 1 FROM knowledge_versions v JOIN knowledge_publications p
                        ON p.version_id=v.version_id WHERE v.version_id=NEW.version_id
                        AND v.plant_id=NEW.plant_id AND v.build_status='checked')
                    BEGIN SELECT RAISE(ABORT,'Active knowledge must be a checked scoped publication'); END""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_version_no_update BEFORE UPDATE ON knowledge_versions
                    BEGIN SELECT RAISE(ABORT,'Knowledge versions are immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_version_no_delete BEFORE DELETE ON knowledge_versions
                    BEGIN SELECT RAISE(ABORT,'Knowledge versions are immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_publication_no_update BEFORE UPDATE ON knowledge_publications
                    BEGIN SELECT RAISE(ABORT,'Knowledge publications are immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS knowledge_publication_no_delete BEFORE DELETE ON knowledge_publications
                    BEGIN SELECT RAISE(ABORT,'Knowledge publications are immutable'); END""",
            ):
                db.execute(statement)

    def build(self, request: KnowledgeBuildRequest, actor: Actor) -> dict:
        try:
            request = KnowledgeBuildRequest.model_validate(request.model_dump())
        except (ValidationError, AttributeError) as exc:
            raise DomainError("invalid_draft", "Invalid knowledge build request") from exc
        plant = request.draft.plant_id
        request_hash = digest(request.model_dump())
        with self.store.transaction() as db:
            self.store.authorize(actor, plant, {"admin", "analyst"}, db)
            existing = db.execute("SELECT * FROM knowledge_versions WHERE owner=? AND request_id=?",
                                  (actor.actor_id, request.request_id)).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise DomainError("idempotency_conflict", "Build request ID reused with different content", 409)
                return self._hydrate(existing, actor, db)
        text = export_owl(request.draft)
        reasoner, classified = _classify(text, request.draft, self.java_path, self.timeout_seconds)
        provenance = [{"report_id": r.report_id, "title": r.title, "synthetic": r.synthetic,
                       "source_sha256": digest(r.model_dump()), "text_sha256": _sha(r.text.encode())}
                      for r in request.reports]
        status = "checked" if reasoner["status"] == "success" and classified is not None else "blocked"
        version_id = new_id("knowledge")
        with self.store.transaction() as db:
            self.store.authorize(actor, plant, {"admin", "analyst"}, db)
            # Recheck after classification: a concurrent idempotent build may have won.
            existing = db.execute("SELECT * FROM knowledge_versions WHERE owner=? AND request_id=?",
                                  (actor.actor_id, request.request_id)).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise DomainError("idempotency_conflict", "Build request ID reused with different content", 409)
                return self._hydrate(existing, actor, db)
            # Serialize immutable artifact writes under the store transaction. Besides avoiding
            # duplicate-build orphans, this prevents Store.write_artifact's create race.
            draft_ref = self.store.write_artifact("knowledge_draft", request.draft.model_dump())
            reports_ref = self.store.write_artifact("original_reports", [r.model_dump() for r in request.reports])
            ontology_ref = self.store.write_artifact("asserted_owl", text)
            classified_ref = self.store.write_artifact("classified_owl", classified) if classified is not None else None
            reasoner_ref = self.store.write_artifact("hermit_report", reasoner)
            manifest = {"schema_version": 1, "draft": draft_ref, "reports": reports_ref,
                        "asserted_owl": ontology_ref, "classified_owl": classified_ref,
                        "reasoner": reasoner_ref, "provenance": provenance}
            content_hash = digest(manifest)
            db.execute("INSERT INTO knowledge_versions VALUES(?,?,?,?,?,?,?,?,?)",
                       (version_id, plant, actor.actor_id, request.request_id, request_hash,
                        content_hash, status, canonical(manifest), now()))
            self.store.audit(db, actor.actor_id, "knowledge_build", {
                "version_id": version_id, "plant_id": plant, "content_hash": content_hash,
                "status": status, "reasoner_status": reasoner["status"], "provenance": provenance})
            row = db.execute("SELECT * FROM knowledge_versions WHERE version_id=?", (version_id,)).fetchone()
            return self._hydrate(row, actor, db)

    def _hydrate(self, row, actor: Actor, db) -> dict:
        self.store.authorize(actor, row["plant_id"], db=db)
        try:
            manifest = json.loads(row["manifest_json"])
            if digest(manifest) != row["content_hash"] or manifest["schema_version"] != 1:
                raise ValueError("Manifest hash mismatch")
            values = {}
            kinds = {"draft": "knowledge_draft", "reports": "original_reports", "asserted_owl": "asserted_owl",
                     "classified_owl": "classified_owl", "reasoner": "hermit_report"}
            for key, kind in kinds.items():
                reference = manifest[key]
                if key == "classified_owl" and reference is None:
                    values[key] = None
                else:
                    if not isinstance(reference, dict) or reference.get("kind") != kind:
                        raise ValueError("Artifact kind mismatch")
                    values[key] = self.store.read_artifact(reference)
            request = KnowledgeBuildRequest(request_id=row["request_id"], draft=values["draft"], reports=values["reports"])
            if request.draft.plant_id != row["plant_id"] or digest(request.model_dump()) != row["request_hash"]:
                raise ValueError("Knowledge scope or request hash mismatch")
            provenance = [{"report_id": r.report_id, "title": r.title, "synthetic": r.synthetic,
                           "source_sha256": digest(r.model_dump()), "text_sha256": _sha(r.text.encode())}
                          for r in request.reports]
            if manifest["provenance"] != provenance:
                raise ValueError("Original report provenance mismatch")
            reasoner = values["reasoner"]
            _check_local_owl(values["asserted_owl"])
            if reasoner["input_owl_sha256"] != _sha(values["asserted_owl"].encode()):
                raise ValueError("Reasoner input hash mismatch")
            if row["build_status"] == "checked":
                if (reasoner.get("status") != "success" or reasoner.get("success") is not True
                        or reasoner.get("consistent") is not True or reasoner.get("unsatisfiable_classes")
                        or not reasoner.get("reload_verified") or not reasoner.get("inferences")
                        or reasoner.get("hermit_exit_code") != 0 or values["classified_owl"] is None):
                    raise ValueError("Missing successful HermiT attestation")
                _check_local_owl(values["classified_owl"])
                if reasoner["classified_owl_sha256"] != _sha(values["classified_owl"].encode()):
                    raise ValueError("Reasoner output hash mismatch")
        except DomainError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            raise DomainError("artifact_integrity", "Knowledge artifact verification failed", 409) from exc
        publication = db.execute("SELECT owner,created_at,review_note FROM knowledge_publications "
                                 "WHERE version_id=? ORDER BY created_at DESC LIMIT 1", (row["version_id"],)).fetchone()
        active = db.execute("SELECT version_id,revision FROM knowledge_active WHERE plant_id=?",
                            (row["plant_id"],)).fetchone()
        return {"version_id": row["version_id"], "plant_id": row["plant_id"],
                "content_hash": row["content_hash"], "status": "published" if publication else row["build_status"],
                "build_status": row["build_status"], "created_at": row["created_at"],
                "draft": values["draft"], "reasoner": reasoner, "provenance": provenance,
                "artifacts": manifest, "active": bool(active and active["version_id"] == row["version_id"]),
                "active_revision": active["revision"] if active else None,
                "publication": dict(publication) if publication else None}

    def get(self, version_id: str, actor: Actor) -> dict:
        with self.store.transaction() as db:
            self.store.current_actor(actor, db)
            row = db.execute("SELECT * FROM knowledge_versions WHERE version_id=?", (version_id,)).fetchone()
            if row is None:
                raise DomainError("knowledge_not_found", "Knowledge version not found", 404)
            return self._hydrate(row, actor, db)

    def active(self, plant: str, actor: Actor, db=None) -> dict:
        """Verify the active publication, optionally within the caller's atomic transaction.

        A supplied connection stays caller-owned: this method does not open another
        transaction, commit, roll back, or close it. Identity and active version are
        therefore checked against the same snapshot as an online task's guarded write.
        """
        if db is None:
            with self.store.transaction() as connection:
                return self.active(plant, actor, db=connection)
        self.store.authorize(actor, plant, db=db)
        row = db.execute("SELECT v.* FROM knowledge_versions v JOIN knowledge_active a "
                         "ON a.version_id=v.version_id WHERE a.plant_id=?", (plant,)).fetchone()
        if row is None:
            raise DomainError("no_active_knowledge", "Plant has no published knowledge", 409)
        value = self._hydrate(row, actor, db)
        if (value["plant_id"] != plant or value["status"] != "published"
                or value["build_status"] != "checked" or not value["active"]):
            raise DomainError("artifact_integrity", "Active knowledge is not a checked scoped publication", 409)
        return value

    def list_versions(self, actor: Actor) -> list:
        with self.store.transaction() as db:
            self.store.current_actor(actor, db)
            rows = db.execute("SELECT * FROM knowledge_versions ORDER BY created_at,version_id").fetchall()
            return [self._hydrate(row, actor, db) for row in rows if row["plant_id"] in actor.plants]

    def publish(self, request: PublishRequest, actor: Actor) -> dict:
        try:
            request = PublishRequest.model_validate(request.model_dump())
        except (ValidationError, AttributeError) as exc:
            raise DomainError("invalid_publication", "Invalid human publication request") from exc
        request_hash = digest(request.model_dump())
        with self.store.transaction() as db:
            self.store.current_actor(actor, db)
            if actor.role != "admin":
                raise DomainError("forbidden", "Human administrator publication is required", 403)
            row = db.execute("SELECT * FROM knowledge_versions WHERE version_id=?", (request.version_id,)).fetchone()
            if row is None:
                raise DomainError("knowledge_not_found", "Knowledge version not found", 404)
            value = self._hydrate(row, actor, db)
            receipt = db.execute("SELECT * FROM knowledge_publications WHERE owner=? AND action_id=?",
                                 (actor.actor_id, request.action_id)).fetchone()
            if receipt:
                if receipt["request_hash"] != request_hash:
                    raise DomainError("idempotency_conflict", "Publication action ID reused with different content", 409)
                return json.loads(receipt["result_json"])
            if request.content_hash != value["content_hash"]:
                raise DomainError("knowledge_hash_mismatch", "Review must name the exact immutable content hash", 409)
            if value["build_status"] != "checked" or value["reasoner"]["status"] != "success":
                raise DomainError("knowledge_not_checked", "Only a real successful HermiT build can be published", 409)
            current = db.execute("SELECT * FROM knowledge_active WHERE plant_id=?", (value["plant_id"],)).fetchone()
            revision = (current["revision"] + int(current["version_id"] != request.version_id)) if current else 1
            stamp = now()
            result = {"version_id": request.version_id, "content_hash": request.content_hash,
                      "plant_id": value["plant_id"], "status": "published", "active_revision": revision,
                      "published_by": actor.actor_id, "published_at": stamp}
            db.execute("INSERT INTO knowledge_publications VALUES(?,?,?,?,?,?,?)",
                       (actor.actor_id, request.action_id, request_hash, request.version_id,
                        request.review_note, canonical(result), stamp))
            db.execute("INSERT INTO knowledge_active VALUES(?,?,?,?) ON CONFLICT(plant_id) DO UPDATE SET "
                       "version_id=excluded.version_id,revision=excluded.revision,activated_at=excluded.activated_at",
                       (value["plant_id"], request.version_id, revision, stamp))
            self.store.audit(db, actor.actor_id, "knowledge_publish", {
                **result, "action_id": request.action_id, "review_note": request.review_note})
            return result


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "--hermit-worker":
        raise SystemExit("This private module entry point only accepts --hermit-worker")
    _hermit_worker(*(Path(arg) for arg in sys.argv[2:]))
