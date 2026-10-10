"""Spec 005 recent runs without Tk: reading each destination's ``runs.tsv`` (T9), and the offers, conversion
outputs and main report read from run records (T10)."""

from datetime import datetime
from pathlib import Path

import pytest

from manager_helpers import CALIBRATION, COMPACT, SPLITS, record_stage as stage, recorded_run as run
from src.petsys_manager.artifacts import RunStore
from src.petsys_manager.contracts import Artifact, CommandResult, InputDescriptor, ResultStatus
from src.petsys_manager.recent import (RecentRun, RunRecordError, TAIL_BYTES, conversion_outputs, list_recent,
                                       main_report, read_overview, run_offers)
from src.petsys_manager.workflow import RUNS, RUNS_HEADER, append_overview

HEADER = "\t".join(RUNS_HEADER) + "\n"


def line(minute, folder=None, status="succeeded"):
    folder = folder or f"run_cal_2026-10-10_12{minute:02d}"
    return f"2026-10-10 12:{minute:02d}:00\t{folder}\tcalibrate\t2 file(s): a.ldat\t{status}\t-\n"


def write(destination, text):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / RUNS).write_bytes(text.encode("utf-8"))
    return destination


def folders(rows):
    return [row.run_root.name for row in rows]


@pytest.mark.fr("005-FR-3")
def test_overview_missing_file_is_no_runs_recorded(tmp_path):
    overview = read_overview(tmp_path)
    assert (overview.rows, overview.skipped, overview.recorded) == ((), 0, False)


@pytest.mark.fr("005-FR-3")
@pytest.mark.parametrize("text", ["", HEADER])
def test_overview_empty_or_header_only_has_no_rows(tmp_path, text):
    overview = read_overview(write(tmp_path, text))
    assert (overview.rows, overview.skipped, overview.recorded) == ((), 0, True)


@pytest.mark.fr("005-FR-3")
def test_overview_row_fields(tmp_path):
    overview = read_overview(write(tmp_path, HEADER + line(5)), "calibration_dir")
    assert overview.rows == (RecentRun(datetime(2026, 10, 10, 12, 5), "calibration_dir",
                                       tmp_path / "run_cal_2026-10-10_1205", "calibrate", "2 file(s): a.ldat",
                                       "succeeded", "-"),)


@pytest.mark.fr("005-FR-3")
def test_overview_partial_last_line_is_skipped_uncounted(tmp_path):
    overview = read_overview(write(tmp_path, HEADER + line(1) + line(2)[:-1]))
    assert (folders(overview.rows), overview.skipped) == (["run_cal_2026-10-10_1201"], 0)


@pytest.mark.fr("005-FR-3")
@pytest.mark.parametrize("bad", ["2026-10-10 12:03:00\tonly\tthree\n",                     # 3 columns
                                 line(3)[:-1] + "\textra\n",                              # 7 columns
                                 line(3, folder=".."), line(3, folder="a/b"),             # not portable
                                 line(3, folder="a\\b"), line(3, folder="-"),
                                 line(3).replace("2026-10-10 12:03:00", "yesterday")])   # no time
def test_overview_malformed_lines_are_skipped_and_counted(tmp_path, bad):
    overview = read_overview(write(tmp_path, HEADER + line(1) + bad + line(4)))
    assert folders(overview.rows) == ["run_cal_2026-10-10_1204", "run_cal_2026-10-10_1201"]
    assert overview.skipped == 1


@pytest.mark.fr("005-FR-3")
def test_overview_reads_only_the_tail_of_a_large_file(tmp_path):
    lines = [f"2026-10-10 12:00:00\trun_{index:07d}\tcalibrate\t{'x' * 60}\tsucceeded\t-\n" for index in range(36_000)]
    text = HEADER + "".join(lines)
    assert len(text) > 3 * 2**20
    overview = read_overview(write(tmp_path, text))
    names = folders(overview.rows)
    assert names[0] == "run_0035999"                     # the newest line is read
    assert overview.skipped == 0                         # the line cut at the tail start isn't counted
    tail_lines = TAIL_BYTES // len(lines[0])
    assert tail_lines - 1 <= len(names) <= tail_lines
    assert TAIL_BYTES == 2**20


@pytest.mark.fr("005-FR-3")
def test_recent_merges_destinations_newest_first_and_reads_duplicates_once(tmp_path):
    data = write(tmp_path / "data", HEADER + line(1, "conv_a") + line(7, "conv_b"))
    cal = write(tmp_path / "cal", HEADER + line(4, "cal_a") + line(9, "cal_b"))
    recent = list_recent([("data_dir", data), ("calibration_dir", cal), ("report_dir", tmp_path / "data" / "."),
                          ("lm_dir", tmp_path / "lm")])
    assert folders(recent.rows) == ["cal_b", "conv_b", "cal_a", "conv_a"]
    assert [row.destination for row in recent.rows] == ["calibration_dir", "data_dir", "calibration_dir", "data_dir"]
    assert recent.unrecorded == ("lm_dir",)
    assert recent.skipped == 0


@pytest.mark.fr("005-FR-3")
def test_recent_same_time_keeps_the_later_line_first(tmp_path):
    data = write(tmp_path / "data", HEADER + line(1, "first") + line(1, "second"))
    assert folders(list_recent([("data_dir", data)]).rows) == ["second", "first"]


@pytest.mark.fr("005-FR-3")
def test_recent_is_capped_at_500_rows_and_sums_skipped(tmp_path):
    data = write(tmp_path / "data", HEADER + "".join(line(minute % 60, f"d{index}") for index, minute in
                                                     enumerate(range(400))) + "bad\n")
    cal = write(tmp_path / "cal", HEADER + "".join(line(minute % 60, f"c{index}") for index, minute in
                                                   enumerate(range(300))) + "bad\n")
    recent = list_recent([("data_dir", data), ("calibration_dir", cal)])
    assert len(recent.rows) == 500
    assert recent.skipped == 2


@pytest.mark.fr("005-FR-3")
def test_overview_parses_lines_written_by_append_overview(tmp_path):
    rows = [("2026-10-10 12:01:02", "conv_2026-10-10_1201", "convert", "a.rawf", "succeeded",
             "conv_coincCompact.ldat (+2 more)"),
            ("2026-10-10 13:14:15", "cal_P5_2026-10-10_1314", "calibrate", "3 file(s): x.ldat", "failed", "-")]
    for row in rows:
        append_overview(tmp_path, row)
    overview = read_overview(tmp_path, "data_dir")
    assert [(row.finished.strftime("%Y-%m-%d %H:%M:%S"), row.run_root.name, row.action, row.inputs, row.status,
             row.main_output) for row in reversed(overview.rows)] == rows
    assert overview.skipped == 0


@pytest.mark.fr("005-FR-3")
def test_reading_changes_nothing(tmp_path):
    data = write(tmp_path / "data", HEADER + line(1))
    before = (data / RUNS).read_bytes(), (data / RUNS).stat().st_mtime_ns
    list_recent([("data_dir", data)])
    assert ((data / RUNS).read_bytes(), (data / RUNS).stat().st_mtime_ns) == before
    assert sorted(path.name for path in Path(data).iterdir()) == [RUNS]


# T10: offers, conversion outputs and the main report from run records ----------------------------------

def offers(root):
    return [(offer.label, offer.target, offer.payload) for offer in run_offers(root)]


@pytest.mark.fr("005-FR-4")
def test_conversion_run_offers_its_recorded_ldats_in_order(tmp_path):
    root = run(tmp_path / "data", ["conversion"], {"conversion": SPLITS})
    (root / "x_coincCompact_3.ldat").write_bytes(b"look-alike")       # not recorded: never offered
    ldats = tuple(InputDescriptor(root / name, *COMPACT, validated=True) for name, _, _ in SPLITS)
    assert offers(root) == [("Calibrate", "calibrate", ldats), ("Generate LM", "listmode", ldats),
                            ("Run QC", "qc_analyze", ldats)]
    assert conversion_outputs(root) == ldats


@pytest.mark.fr("005-FR-4")
def test_recorded_file_that_is_not_an_output_is_never_offered(tmp_path):
    destination = tmp_path / "data"
    destination.mkdir()
    store = RunStore.reserve(destination, {"action": "convert"}, name="conv", stages=["conversion"])
    attempt = store.reserve_attempt("conversion", attempt_id="attempt-1")
    extra = attempt.directory / "x_coincCompact_9.ldat"           # recorded during the run, not a result output
    extra.write_bytes(b"partial")
    store.record_artifacts(attempt, [Artifact(extra, "ldat", InputDescriptor(extra, *COMPACT))])
    artifacts = []
    for name, kind, content in SPLITS:
        (attempt.directory / name).write_bytes(content)
        artifacts.append(Artifact(attempt.directory / name, kind,
                                  InputDescriptor(attempt.directory / name, *COMPACT, validated=True)))
    store.finish_attempt(attempt, CommandResult(attempt.identity, ResultStatus.SUCCEEDED, 0, "ok", artifacts,
                                                outputs_validated=True))
    store.finish(ResultStatus.SUCCEEDED)
    assert [item.path.name for item in conversion_outputs(store.root)] == [name for name, _, _ in SPLITS]


@pytest.mark.fr("005-FR-4")
def test_calibration_run_offers_its_encal(tmp_path):
    root = run(tmp_path / "cal", ["calibration"], {"calibration": CALIBRATION})
    assert offers(root) == [("Generate LM with this calibration", "lm_calibration", root / "x.encal")]


@pytest.mark.fr("005-FR-4")
def test_pipeline_run_offers_both_kinds(tmp_path):
    root = run(tmp_path / "data", ["conversion", "calibration"], {"conversion": SPLITS, "calibration": CALIBRATION})
    assert [(label, target) for label, target, _ in offers(root)] == [
        ("Calibrate", "calibrate"), ("Generate LM", "listmode"), ("Run QC", "qc_analyze"),
        ("Generate LM with this calibration", "lm_calibration")]
    assert offers(root)[-1][2] == root / "2_calibration" / "x.encal"
    assert [item.path.parent.name for item in offers(root)[0][2]] == ["1_conversion"] * 3


@pytest.mark.fr("005-FR-4")
@pytest.mark.parametrize("status", [ResultStatus.FAILED, ResultStatus.CANCELLED])
def test_unsuccessful_stage_gives_no_offer(tmp_path, status):
    root = run(tmp_path / "data", ["conversion", "calibration"],
               {"conversion": SPLITS, "calibration": (CALIBRATION, status)})
    assert [target for _, target, _ in offers(root)] == ["calibrate", "listmode", "qc_analyze"]
    failed = run(tmp_path / "cal", ["calibration"], {"calibration": (CALIBRATION, status)})
    assert offers(failed) == []


@pytest.mark.fr("005-FR-4")
@pytest.mark.parametrize("change", ["resize", "remove"])
def test_changed_output_is_refused_naming_the_file(tmp_path, change):
    root = run(tmp_path / "data", ["conversion"], {"conversion": SPLITS})
    target = root / "x_coincCompact_1.ldat"
    target.write_bytes(b"short") if change == "resize" else target.unlink()
    for read in (run_offers, conversion_outputs):
        with pytest.raises(RunRecordError, match="x_coincCompact_1.ldat"):
            read(root)


@pytest.mark.fr("005-FR-4")
def test_changed_calibration_is_refused_naming_the_file(tmp_path):
    root = run(tmp_path / "cal", ["calibration"], {"calibration": CALIBRATION})
    (root / "x.encal").write_bytes(b"edited by hand")
    with pytest.raises(RunRecordError, match="x.encal"):
        run_offers(root)


@pytest.mark.fr("005-FR-6")
def test_pre_t29_run_folder_still_reads(tmp_path):
    destination = tmp_path / "data"
    destination.mkdir()
    store = RunStore.reserve(destination, {"action": "convert"}, run_id="old_run")    # T4 layout: <stage>/<attempt>/
    directory = stage(store, "conversion", SPLITS)
    store.finish(ResultStatus.SUCCEEDED)
    root = store.root
    for revision in (root / ".history").iterdir():          # revisions in the run folder, as before T29
        revision.rename(root / revision.name)
    (root / ".history").rmdir()
    assert conversion_outputs(root) == tuple(InputDescriptor(directory / name, *COMPACT, validated=True)
                                             for name, _, _ in SPLITS)


@pytest.mark.fr("005-FR-6")
def test_non_run_folder_and_run_without_conversion_outputs_are_refused(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x_coincCompact_0.ldat").write_bytes(b"x")
    for read in (run_offers, conversion_outputs, main_report):
        with pytest.raises(RunRecordError, match="not a run folder"):
            read(plain)
    calibration = run(tmp_path / "cal", ["calibration"], {"calibration": CALIBRATION})
    with pytest.raises(RunRecordError, match="no conversion outputs"):
        conversion_outputs(calibration)


@pytest.mark.fr("005-FR-3")
def test_main_report_per_action(tmp_path):
    qc = run(tmp_path / "qc", ["qc"], {"qc": [("report.pdf", "qc_report", b"pdf"), ("summary.json", "qc_summary", b"{}")]})
    calibration = run(tmp_path / "cal", ["calibration"], {"calibration": CALIBRATION})
    lm = run(tmp_path / "lm", ["listmode"], {"listmode": [("x.lm", "listmode", b"lm"),
                                                         ("x_debug_a.png", "listmode_debug_plot", b"a"),
                                                         ("x_debug_b.png", "listmode_debug_plot", b"b")]})
    lm_plain = run(tmp_path / "lm2", ["listmode"], {"listmode": [("x.lm", "listmode", b"lm")]})
    conversion = run(tmp_path / "data", ["conversion"], {"conversion": SPLITS})
    assert main_report(qc) == qc / "report.pdf"
    assert main_report(calibration) == calibration / "x_plot.png"
    assert main_report(lm) == lm / "x_debug_a.png"
    assert main_report(lm_plain) is None
    assert main_report(conversion) is None


@pytest.mark.fr("005-FR-3")
def test_main_report_of_a_pipeline_is_its_last_stage_report(tmp_path):
    root = run(tmp_path / "data", ["conversion", "calibration", "listmode"],
               {"conversion": SPLITS, "calibration": CALIBRATION,
                "listmode": [("x.lm", "listmode", b"lm"), ("x_debug.png", "listmode_debug_plot", b"d")]})
    assert main_report(root) == root / "3_listmode" / "x_debug.png"
