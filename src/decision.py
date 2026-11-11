"""Assignment-level experiment decisions with explicit unavailable evidence and atomic releases."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .abtest import srm_check, two_prop_test


def validate_assignments(frame: pd.DataFrame, arms=("gate_30", "gate_40"), metrics=("retention_7", "retention_1")):
    required = {"userid", "version", *metrics}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("nonempty assignments require userid, version and all registered metrics")
    ids = frame.userid.astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("assignment identities must be nonempty and unique; do not silently deduplicate")
    if not frame.version.isin(arms).all() or set(frame.version) != set(arms):
        raise ValueError("exactly the two declared arms must be represented")
    for metric in metrics:
        if not frame[metric].isin([0, 1, False, True]).all():
            raise ValueError(f"{metric}: binary outcomes must be observed for every assigned participant")
    if "assigned_at" in frame:
        times = pd.to_datetime(frame.assigned_at, utc=True, errors="coerce")
        if times.isna().any():
            raise ValueError("assignment timestamps must be valid when supplied")


def exposure_audit(assignments, exposures):
    """Exposures are diagnostics; they never change the intention-to-treat denominator."""
    if exposures is None:
        return {"status": "unavailable", "reason": "source has no exposure-event log", "exposed_participants": None}
    if "assigned_at" not in assignments or not {"event_id", "userid", "version", "exposed_at"} <= set(exposures):
        raise ValueError("exposure validation requires assignment times and event_id/userid/version/exposed_at")
    if exposures.event_id.isna().any() or exposures.event_id.astype(str).str.strip().eq("").any() or exposures.event_id.duplicated().any():
        raise ValueError("exposure event identities must be nonempty and unique")
    joined = exposures.merge(assignments[["userid", "version", "assigned_at"]], on="userid", how="left",
                             suffixes=("_exposed", "_assigned"), validate="many_to_one", indicator=True)
    if (joined._merge != "both").any() or (joined.version_exposed != joined.version_assigned).any():
        raise ValueError("exposures must belong to a known participant and their assigned arm")
    exposed = pd.to_datetime(joined.exposed_at, utc=True, errors="coerce")
    assigned = pd.to_datetime(joined.assigned_at, utc=True, errors="coerce")
    if exposed.isna().any() or (exposed < assigned).any():
        raise ValueError("exposures must have valid timestamps at or after assignment")
    return {"status": "observed", "events": len(exposures), "exposed_participants": int(exposures.userid.nunique()),
            "unexposed_participants": int(len(assignments) - exposures.userid.nunique()), "denominator": "all assigned participants"}


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    primary, guards = protocol.get("primary"), protocol.get("guardrails")
    if not isinstance(primary, str) or not primary or not isinstance(guards, dict) or primary in guards:
        raise ValueError("one primary and distinct guardrail metrics are required")
    for value in [protocol.get("alpha"), protocol.get("srm_alpha"), protocol.get("expected_control_share")]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError("alpha and allocation parameters must be between zero and one")
    for value in guards.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError("guardrail margins must be finite absolute rate declines in [0, 1)")
    if protocol.get("registration") not in {"retrospective", "declared_preregistered"}:
        raise ValueError("registration must explicitly disclose retrospective or declared_preregistered status")
    if protocol["registration"] == "declared_preregistered":
        registered = pd.to_datetime(protocol.get("registered_at"), utc=True, errors="coerce")
        cutoff = pd.to_datetime(protocol.get("analysis_cutoff"), utc=True, errors="coerce")
        if pd.isna(registered) or pd.isna(cutoff) or registered >= cutoff:
            raise ValueError("declared preregistration must precede the analysis cutoff")
    if "analysis_cutoff" in protocol and pd.isna(pd.to_datetime(protocol["analysis_cutoff"], utc=True, errors="coerce")):
        raise ValueError("analysis_cutoff must be a valid timestamp")
    arms = protocol.get("arms")
    if not isinstance(arms, list) or len(arms) != 2 or len(set(arms)) != 2 or any(not isinstance(a, str) or not a for a in arms):
        raise ValueError("two distinct arm names are required")


def decide(assignments, protocol, exposures=None):
    validate_protocol(protocol)
    metrics = [protocol["primary"], *protocol["guardrails"]]
    validate_assignments(assignments, protocol["arms"], metrics)
    exposure = exposure_audit(assignments, exposures)
    if "analysis_cutoff" in protocol:
        cutoff = pd.to_datetime(protocol["analysis_cutoff"], utc=True)
        if "assigned_at" in assignments and (pd.to_datetime(assignments.assigned_at, utc=True) > cutoff).any():
            raise ValueError("assignments after the declared analysis cutoff are not eligible")
        if exposures is not None and (pd.to_datetime(exposures.exposed_at, utc=True) > cutoff).any():
            raise ValueError("exposures after the declared analysis cutoff are not eligible")
    a, b = [assignments[assignments.version == arm] for arm in protocol["arms"]]
    srm = srm_check(len(a), len(b), protocol["expected_control_share"])
    # Bonferroni simultaneous two-sided intervals cover the registered family.
    tests = {metric: two_prop_test(int(a[metric].sum()), len(a), int(b[metric].sum()), len(b),
                                  alpha=protocol["alpha"] / len(metrics)) for metric in metrics}
    primary = tests[protocol["primary"]]
    guards = {name: bool(tests[name]["ci_low"] is not None and tests[name]["ci_low"] >= -margin)
              for name, margin in protocol["guardrails"].items()}
    if srm < protocol["srm_alpha"]:
        decision = "investigate_assignment_integrity"
    elif primary["ci_high"] is not None and primary["ci_high"] < 0:
        decision = "retain_control_primary_harm"
    elif primary["ci_low"] is not None and primary["ci_low"] > 0 and all(guards.values()):
        decision = "treatment_supported_under_declared_rule"
    else:
        decision = "inconclusive_retain_control"
    if protocol["registration"] == "declared_preregistered" and "assigned_at" in assignments:
        registered = pd.to_datetime(protocol["registered_at"], utc=True)
        if registered >= pd.to_datetime(assignments.assigned_at, utc=True).min():
            raise ValueError("declared preregistration must precede the first supplied assignment")
    return {"schema_version": 1, "protocol": protocol, "registration_verified": False,
            "decision": decision, "scope": "retrospective illustration" if protocol["registration"] == "retrospective" else "declared preregistration; not independently verified",
            "participants": len(assignments), "arm_counts": dict(zip(protocol["arms"], [len(a), len(b)])),
            "denominator": "intention to treat: one row per assigned participant; no exposure filtering",
            "srm": {"p_value": srm, "alarm_threshold": protocol["srm_alpha"], "warning_below_0_05": srm < .05},
            "metrics": tests, "guardrails_pass": guards, "interval_method": "Bonferroni simultaneous Wald intervals; sparse or constant arms require another design",
            "exposure": exposure, "assignment_timestamps": "observed" if "assigned_at" in assignments else "unavailable",
            "unavailable": ["pre-experiment covariate/CUPED", "observation-window maturity without event timestamps", "independent randomization audit"],
            "limits": ["The protocol for Cookie Cats was specified retrospectively after published results were known.",
                       "Missing exposure/timestamps prevent exposure, novelty, maturity and temporal integrity verification.",
                       "No rollout, business impact or independent human review is claimed."]}


def publish(report, destination: Path):
    """Publish a new immutable evidence directory; failure preserves all prior releases."""
    if destination.exists():
        raise FileExistsError("choose a new release directory; existing evidence is immutable")
    encoded = json.dumps(report, indent=2, allow_nan=False).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".decision-", dir=destination.parent))
    try:
        (stage / "report.json").write_bytes(encoded)
        page = ("<!doctype html><meta charset='utf-8'><title>Experiment decision</title>"
                "<style>body{font:16px system-ui;max-width:980px;margin:3rem auto;padding:1rem}pre{white-space:pre-wrap}</style>"
                f"<h1>{html.escape(report['decision'].replace('_', ' '))}</h1><p>{html.escape(report['scope'])}</p>"
                f"<pre>{html.escape(encoded.decode())}</pre>")
        (stage / "index.html").write_text(page, encoding="utf-8")
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cookie_cats.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("protocols/cookie-cats.json"))
    parser.add_argument("--exposures", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, protocol_bytes = args.data.read_bytes(), args.protocol.read_bytes()
    exposure_bytes = args.exposures.read_bytes() if args.exposures else None
    report = decide(pd.read_csv(io.BytesIO(source)), json.loads(protocol_bytes),
                    pd.read_csv(io.BytesIO(exposure_bytes)) if exposure_bytes is not None else None)
    report["provenance"] = {"source_sha256": hashlib.sha256(source).hexdigest(),
                            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "inference_code_sha256": hashlib.sha256(Path(__file__).with_name("abtest.py").read_bytes()).hexdigest(),
                            "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    if args.exposures:
        report["provenance"]["exposure_sha256"] = hashlib.sha256(exposure_bytes).hexdigest()
    publish(report, args.out)
    print(json.dumps({"decision": report["decision"], "participants": report["participants"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
