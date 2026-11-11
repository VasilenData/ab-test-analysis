import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.decision import decide, exposure_audit, publish, validate_assignments


def protocol():
    return json.loads(Path("protocols/cookie-cats.json").read_text())


def assignments():
    return pd.DataFrame({"userid": range(400), "version": ["gate_30"] * 200 + ["gate_40"] * 200,
                         "retention_7": [1] * 100 + [0] * 100 + [1] * 50 + [0] * 150,
                         "retention_1": [1] * 150 + [0] * 50 + [1] * 150 + [0] * 50})


@pytest.mark.parametrize("mutation", [
    lambda x: pd.concat([x, x.iloc[[0]]]),
    lambda x: x.assign(version=["other"] + x.version.tolist()[1:]),
    lambda x: x.assign(retention_7=[None] + x.retention_7.tolist()[1:]),
    lambda x: x.assign(userid=[None] + x.userid.tolist()[1:]),
])
def test_assignment_contract_stops_denominator_and_identity_corruption(mutation):
    with pytest.raises(ValueError):
        validate_assignments(mutation(assignments()))


def test_hand_computed_lift_and_missing_evidence_remain_visible():
    report = decide(assignments(), protocol())
    assert report["metrics"]["retention_7"]["diff"] == -.25
    assert report["arm_counts"] == {"gate_30": 200, "gate_40": 200}
    assert report["decision"] == "retain_control_primary_harm"
    assert report["exposure"]["status"] == "unavailable"
    assert report["assignment_timestamps"] == "unavailable"
    assert not report["registration_verified"]
    assert decide(assignments().sample(frac=1, random_state=4), protocol()) == report


def test_exposure_diagnostics_do_not_condition_primary_analysis_on_exposure():
    data = assignments().assign(assigned_at="2026-01-01T00:00:00Z")
    events = pd.DataFrame({"event_id": ["a", "b"], "userid": [0, 0], "version": ["gate_30"] * 2,
                           "exposed_at": ["2026-01-02T00:00:00Z"] * 2})
    report = decide(data, protocol(), events)
    assert report["participants"] == 400 and report["exposure"]["exposed_participants"] == 1
    assert report["exposure"]["unexposed_participants"] == 399
    assert report["metrics"] == decide(data, protocol())["metrics"]
    for bad in [events.assign(userid=[999, 0]), events.assign(version=["gate_40"] * 2),
                events.assign(exposed_at="2025-01-01"), events.assign(event_id="a")]:
        with pytest.raises(ValueError):
            exposure_audit(data, bad)


def test_preregistration_claim_cannot_postdate_assignments():
    config = protocol() | {"registration": "declared_preregistered", "registered_at": "2026-02-01",
                           "analysis_cutoff": "2026-03-01"}
    with pytest.raises(ValueError, match="first supplied assignment"):
        decide(assignments().assign(assigned_at="2026-01-01"), config)


def test_srm_failure_blocks_positive_decision():
    data = assignments()
    data = data[data.userid < 220]
    assert decide(data, protocol())["decision"] == "investigate_assignment_integrity"


def test_immutable_release_and_interruption_preserve_previous_evidence(tmp_path, monkeypatch):
    from src import decision
    report = decide(assignments(), protocol())
    old, new = tmp_path / "old", tmp_path / "new"
    publish(report, old)
    before = (old / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish(report, old)
    def fail(*args):
        raise OSError("injected rename interruption")
    monkeypatch.setattr(decision.os, "rename", fail)
    with pytest.raises(OSError):
        publish(report, new)
    assert not new.exists() and (old / "report.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [old]


def test_nonfinite_protocol_rejected_before_analysis():
    config = copy.deepcopy(protocol())
    config["guardrails"]["retention_1"] = float("nan")
    with pytest.raises(ValueError):
        decide(assignments(), config)


def test_declared_analysis_cutoff_excludes_future_assignments_and_exposures():
    config = protocol() | {"registration": "declared_preregistered", "registered_at": "2025-12-01",
                           "analysis_cutoff": "2026-01-10"}
    data = assignments().assign(assigned_at="2026-01-20")
    with pytest.raises(ValueError, match="assignments after"):
        decide(data, config)
    data = assignments().assign(assigned_at="2026-01-10")
    assert decide(data, config)["participants"] == 400
    events = pd.DataFrame({"event_id": ["one"], "userid": [0], "version": ["gate_30"],
                           "exposed_at": ["2026-01-11"]})
    with pytest.raises(ValueError, match="exposures after"):
        decide(data, config, events)
    events["exposed_at"] = "2026-01-10"
    assert decide(data, config, events)["exposure"]["exposed_participants"] == 1
