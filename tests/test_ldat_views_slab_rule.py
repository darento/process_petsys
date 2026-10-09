"""The Cornell slab rule (FR-19) in LDATInspector's GUI and report (spec 002 T16; spec 007 T22).

The selector state, the Settings it builds, the SM summary, log and plot labels, and the report
provenance, on the recovery fixtures (``ldat_helpers.RECOVERY_CASES``).

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from pathlib import Path
import tempfile

import pytest

from ldat_helpers import CONFIGS, RECOVERY_CASES, Checks, build, destroy, require_display, side, write_ldat
from src.ldat_inspector.engine import Selection, Settings, load_setup, merge_results, process_file

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def slab_rule_dataset(tmp, rule):
    """ldat_scale_check's FR-19 recovery fixtures read with ``rule``: 88 pairs, SM 0 (det1) and SM 1."""
    from src.ldat_inspector.engine import load_setup, merge_results, process_file

    settings = Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, min_channels=4,
                        min_channel_energy=0.2, calibrated=False, slab_rule=rule)
    setup = load_setup(settings)
    full = [30.0, 20.0, 10.0, 5.0]
    rows = [case for case in RECOVERY_CASES for _ in range(case[1])]
    path = tmp / f"slab_rule_{rule}.ldat"
    write_ldat(path, [(side(setup, 0, 0, case[2], full, 10 ** 12 + i * 5_000_000),
                        side(setup, 1, 0, {3: 12.0, 2: 5.0}, full, 10 ** 12 + i * 5_000_000 + 99))
                       for i, case in enumerate(rows)])
    return merge_results(settings, [process_file(str(path), settings, 0)], setup)


SLAB_RULE_RECOVERED = 76  # RECOVERY_CASES rows without a legacy slab (88 pairs, 12 kept by legacy)


def slab_rule_gui_checks(check, tmp):

    import exe_programs.ldat_inspector_gui as gui

    recover, legacy = (slab_rule_dataset(tmp, rule) for rule in ("recover_non_adjacent", "legacy"))
    check("T16 fixture: 76 recovered SM 0 sides under the recover rule, none under legacy",
          int(recover.modules[0].recovered_slab.sum()) == SLAB_RULE_RECOVERED and len(recover.modules[0]) == 88
          and not recover.modules[1].recovered_slab.any() and len(legacy.modules[0]) == 12
          and not legacy.modules[0].recovered_slab.any())
    app = gui.LDATWorkbench()
    app.withdraw()
    try:
        app.tabs.set(gui.SM_TAB)
        app.system.set("IMAS")
        imas_state = app.slab_rule_combo.cget("state")
        app.system.set("CORNELL")
        check("slab rule selector: Legacy (reject) by default, Cornell only",
              app.slab_rule.get() == "Legacy (reject)" and imas_state == "disabled"
              and app.slab_rule_combo.cget("state") == "readonly")
        app.config_path, app.calibration_path = str(CONFIGS["CORNELL"]), ""
        app.calibrated.set(False)
        default = app._current_settings().slab_rule
        app.slab_rule.set("Recover non-adjacent")
        chosen = app._current_settings().slab_rule
        app.system.set("IMAS")
        imas = app._current_settings().slab_rule
        app.system.set("CORNELL")
        check("Process settings carry the chosen rule (IMAS always legacy)",
              (default, chosen, imas) == ("legacy", "recover_non_adjacent", "legacy"), f"{default}, {chosen}, {imas}")

        app.dataset = recover
        app._update_after_processing()
        app.update()
        summary = app.sm_summary.get("1.0", "end")
        log = app.console.get("1.0", "end")
        check("recover dataset: SM summary shows the rule and SM 0's recovered sides; the log the totals",
              "  Slab rule       recover non-adjacent\n  Recovered sides          76  (86.4 % of SM; "
              "legacy-built keV/limits)" in summary
              and "Slab rule: recover non-adjacent (FR-19); 76 recovered sides (43.2 % of sides). keV calibration "
                  "and limits files built under the legacy rule may be inconsistent" in log
              and app.status.cget("text").startswith("Merged 88 coincidence pairs"), summary.replace("\n", " / "))
        check("plot label: recover rule with the recovered-side count (SM 0 and all SMs)",
              gui._slab_rule_note(recover, {0}) == "recover rule: 76 recovered sides"
              and gui._slab_rule_note(recover, {1}).endswith(": 0 recovered sides")
              and gui._slab_rule_note(legacy) == "legacy rule: 76 unresolved pairs rejected")
        app.slab_rule.set("Legacy (reject)")
        app._slab_rule_changed()
        check("changing the rule with data loaded says it applies on the next Process",
              "applies when the files are processed again" in app.status.cget("text"), app.status.cget("text"))
        app.dataset = legacy
        app._update_after_processing()
        app.update()
        check("legacy dataset: SM summary reads 'legacy (non-adjacent rejected)', no recovered line",
              "Slab rule       legacy (non-adjacent rejected)" in app.sm_summary.get("1.0", "end")
              and "Recovered sides" not in app.sm_summary.get("1.0", "end")
              and "76 unresolved-slab pairs rejected at ingest" in app.console.get("1.0", "end"))
    finally:
        if app._estimate_after is not None:  # the rule selector schedules a memory estimate
            app.after_cancel(app._estimate_after)
        destroy(app)


def slab_rule_report_checks(check, tmp):
    import re

    from pypdf import PdfReader

    from src.ldat_inspector.report import _provenance, write_report

    recover, legacy = (slab_rule_dataset(tmp, rule) for rule in ("recover_non_adjacent", "legacy"))
    selection = Selection(0.0, 1e9)
    pdf = tmp / "recover.pdf"
    write_report(str(pdf), recover, selection, sm=0)
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() for page in PdfReader(str(pdf)).pages))
    check("PDF provenance: recover rule, SM 0's recovered sides and the legacy-calibration warning",
          "Slab rule: recover non-adjacent (FR-19); 76 recovered sides (86.4 % of sides)" in text
          and "built under the legacy rule may be inconsistent for recovered sides (SM 0)" in text, text[:300])
    lines = _provenance(recover, selection) + _provenance(legacy, selection)
    imas = _provenance(build("IMAS")[0], selection)
    check("provenance: whole-system recover and legacy lines; none for IMAS",
          any(line.startswith("Slab rule: recover non-adjacent (FR-19); 76 recovered sides (43.2 % of sides)")
              for line in lines)
          and "Slab rule: legacy (non-adjacent rejected); 76 unresolved-slab pairs rejected at ingest" in lines
          and not any(line.startswith("Slab rule") for line in imas))


SLAB_RULE_GUI = {
    "fixture": "T16 fixture: 76 recovered SM 0 sides under the recover rule, none under legacy",
    "selector": "slab rule selector: Legacy (reject) by default, Cornell only",
    "settings": "Process settings carry the chosen rule (IMAS always legacy)",
    "recover-summary":
        "recover dataset: SM summary shows the rule and SM 0's recovered sides; the log the totals",
    "plot-label": "plot label: recover rule with the recovered-side count (SM 0 and all SMs)",
    "rule-change": "changing the rule with data loaded says it applies on the next Process",
    "legacy-summary": "legacy dataset: SM summary reads 'legacy (non-adjacent rejected)', no recovered line",
}
SLAB_RULE_REPORT = {
    "pdf-provenance":
        "PDF provenance: recover rule, SM 0's recovered sides and the legacy-calibration warning",
    "provenance-lines": "provenance: whole-system recover and legacy lines; none for IMAS",
}


@pytest.fixture(scope="module")
def slab_rule_gui():
    require_display()
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(SLAB_RULE_GUI).run(slab_rule_gui_checks, Path(temporary))


@pytest.mark.gui
@fr("002-FR-19")
@pytest.mark.parametrize("check", SLAB_RULE_GUI)
def test_slab_rule_gui(slab_rule_gui, check):
    slab_rule_gui.verdict(check)


@pytest.fixture(scope="module")
def slab_rule_report():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(SLAB_RULE_REPORT).run(slab_rule_report_checks, Path(temporary))


@fr("002-FR-14", "002-FR-19")
@pytest.mark.parametrize("check", SLAB_RULE_REPORT)
def test_slab_rule_report(slab_rule_report, check):
    slab_rule_report.verdict(check)
