"""Original synthetic conveyor reports and their derived, unpublished knowledge request.

These reports and thresholds are invented demonstration material, not production advice.
"""
from __future__ import annotations

import yaml

from .knowledge import request_from_reports
from .schemas import KnowledgeBuildRequest, ReportSource

DECISION_LABELS = {
    "jam": "Conveyor obstruction / jam",
    "slip": "Drive-to-belt traction slip",
    "sensor_fault": "Encoder sensing discrepancy",
    "normal": "No fault supported in the observed window",
}
TREE_PATH_EXAMPLES = {
    "jam": ["root", "mechanical", "jam"],
    "slip": ["root", "mechanical", "slip"],
    "sensor_fault": ["root", "sensing", "sensor_fault"],
    "normal": ["root", "normal"],
}


def original_reports(plant_id: str = "PLANT-A") -> list[ReportSource]:
    """Three new local source documents with prose and auditable structured fragments."""
    def numeric(condition_id, label, signal, operator, threshold, unit, **extra):
        return {"condition_id": condition_id, "label": label, "source": "signal",
                "signal": signal, "operator": operator, "threshold": threshold, "unit": unit,
                **extra}

    def human(condition_id, label, key, value=True):
        return {"condition_id": condition_id, "label": label, "source": "human",
                "human_key": key, "operator": "eq", "threshold": value, "unit": "bool"}

    def leaf(node_id, conditions, mandatory=(), terms=(), kind="diagnosis"):
        return {"node_id": node_id, "label": DECISION_LABELS[node_id], "kind": kind,
                "gate": "all", "conditions": conditions, "mandatory": list(mandatory),
                "terms": list(terms)}

    fragments = [
        (
            "ORIGINAL-R01", "Synthetic conveyor rule organization and reference operation",
            "Original invented bench memo for a low-energy demonstration conveyor. The operator "
            "first verifies a numeric drive-enabled signal across the complete observation window. "
            "Mechanical and sensing investigations are separate branches. A stable reference run "
            "has current 2 to 6 A, belt speed 0.8 to 1.2 m/s, and independently measured belt, "
            "roller and encoder speeds agreeing within 0.1 m/s. All conditions require at least "
            "20 samples with gaps no larger than 1500 ms. These screening thresholds describe "
            "this fictional demonstration only and do not prove a physical cause.",
            {
                "tree_id": "original-conveyor", "plant_id": plant_id,
                "title": "Original synthetic conveyor evidence tree", "root_id": "root",
                "global_required": ["drive_enabled"],
                "nodes": [
                    {"node_id": "root", "label": "Conveyor investigation", "kind": "branch",
                     "children": ["mechanical", "sensing", "normal"]},
                    {"node_id": "mechanical", "label": "Mechanical transmission", "kind": "branch",
                     "children": ["jam", "slip"]},
                    {"node_id": "sensing", "label": "Sensing and measurement", "kind": "branch",
                     "children": ["sensor_fault"]},
                    leaf("normal", ["current_nominal", "belt_nominal", "roller_aligned", "encoder_aligned"],
                         terms=("normal", "stable", "reference"), kind="normal"),
                ],
                "conditions": [
                    numeric("drive_enabled", "Drive remained enabled", "drive_enabled", "eq", 1.0,
                            "bool", statistic="min"),
                    numeric("current_nominal", "Reference motor current", "motor_current", "between", 2.0,
                            "A", upper=6.0),
                    numeric("belt_nominal", "Reference belt speed", "belt_speed", "between", 0.8,
                            "m/s", upper=1.2),
                    numeric("roller_aligned", "Roller and belt speed agree", "roller_speed", "between", -0.1,
                            "m/s", upper=0.1, statistic="difference_mean", other_signal="belt_speed"),
                    numeric("encoder_aligned", "Encoder and belt speed agree", "encoder_speed", "between", -0.1,
                            "m/s", upper=0.1, statistic="difference_mean", other_signal="belt_speed"),
                ],
            },
        ),
        (
            "ORIGINAL-R02", "Synthetic jam and traction inspection trial",
            "Original invented obstruction trial: an enabled conveyor drew more than 8 A while "
            "the actual belt speed remained below 0.15 m/s. A trained human, after following the "
            "site's safe isolation procedures, confirmed an obstruction. Current and speed alone "
            "do not establish a jam; a boolean obstruction_confirmed human observation is mandatory. "
            "In a separate invented slip trial, roller speed exceeded actual belt speed by more "
            "than 0.25 m/s while current was in the reference band, and a human confirmed traction "
            "loss during a safe inspection. Missing observations leave a diagnosis unconfirmed. "
            "No automated actuation or unsafe inspection is authorized by this document.",
            {
                "nodes": [
                    leaf("jam", ["current_high", "belt_low"], ["obstruction_confirmed"],
                         ("jam", "obstruction", "stopped", "high current")),
                    leaf("slip", ["traction_difference", "current_nominal"], ["traction_loss_confirmed"],
                         ("slip", "traction", "roller", "belt")),
                ],
                "conditions": [
                    numeric("current_high", "Actual motor current is high", "motor_current", "gt", 8.0, "A"),
                    numeric("belt_low", "Actual belt speed is very low", "belt_speed", "lt", 0.15, "m/s"),
                    numeric("traction_difference", "Roller exceeds belt speed", "roller_speed", "gt", 0.25,
                            "m/s", statistic="difference_mean", other_signal="belt_speed"),
                    human("obstruction_confirmed", "Safe human inspection confirms obstruction", "obstruction_confirmed"),
                    human("traction_loss_confirmed", "Safe human inspection confirms traction loss", "traction_loss_confirmed"),
                ],
            },
        ),
        (
            "ORIGINAL-R03", "Synthetic encoder discrepancy comparison",
            "Original invented sensing trial: an enabled conveyor had reference motor current "
            "and an independently measured belt speed within 0.8 to 1.2 m/s, but the encoder "
            "reported more than 0.6 m/s above the independent belt measurement. A human confirmed "
            "that the independent reference was valid after a safe measurement review. This "
            "specific positive discrepancy supports the synthetic encoder-fault leaf; it is "
            "not a universal sensor-failure detector. The classifier checks the consistency "
            "of rule roles and evidence bindings, not the causal truth of this fictional trial.",
            {
                "nodes": [leaf("sensor_fault", ["encoder_discrepancy", "belt_nominal", "current_nominal"],
                               ["independent_measurement_confirmed"],
                               ("encoder", "sensor", "measurement", "discrepancy"))],
                "conditions": [
                    numeric("encoder_discrepancy", "Encoder exceeds independent belt measurement", "encoder_speed",
                            "gt", 0.6, "m/s", statistic="difference_mean", other_signal="belt_speed"),
                    human("independent_measurement_confirmed", "Human confirms independent measurement validity",
                          "independent_measurement_confirmed"),
                ],
            },
        ),
    ]
    return [ReportSource(report_id=report_id, title=title,
                         text=yaml.safe_dump({"summary": summary, "knowledge": fragment},
                                             sort_keys=False, allow_unicode=True))
            for report_id, title, summary, fragment in fragments]


def original_knowledge(plant_id: str = "PLANT-A") -> KnowledgeBuildRequest:
    """Derive an unpublished request from the three original source documents, offline."""
    return request_from_reports("original-knowledge-" + plant_id, original_reports(plant_id))
