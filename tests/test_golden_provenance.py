"""Golden files and their provenance (spec 007 FR-3, T8).

Each file under tests/data/golden/ is a reference oracle's output on the manager_helpers fixtures,
frozen once by the local scripts/capture_golden_007.py; tests compare the Manager with it and never
load a reference. A golden file changes only in a commit citing the requirement or bug id.
"""

import ast
from datetime import date
import hashlib
import json
import re

import pytest

import manager_helpers
from helpers import DATA, REPO

GOLDEN = DATA / "golden"
ORACLES = {"calibration": "scripts_cornell/cornell_slab_en_cal.py",
           "listmode": "scripts_cornell/cornell_listmode_cog_fixed_position.py",
           "qc": "scripts_cornell/cornell_system_validation.py"}
FILES = sorted(p for p in GOLDEN.rglob("*") if p.is_file() and not p.name.endswith(".provenance.json"))
RECORDS = sorted(GOLDEN.rglob("*.provenance.json"))
SHA256 = re.compile(r"[0-9a-f]{64}")
FORBIDDEN = {"scripts", "scripts_cornell", "scripts_imas", "gui_cornell"}

pytestmark = pytest.mark.fr("007-FR-3")


def name(path):
    return path.relative_to(GOLDEN).as_posix()


def test_each_reference_area_has_golden_files():
    assert {path.relative_to(GOLDEN).parts[0] for path in FILES} == set(ORACLES)


@pytest.mark.parametrize("path", FILES, ids=name)
def test_golden_file_has_matching_provenance(path):
    record = json.loads(path.with_name(f"{path.name}.provenance.json").read_text(encoding="utf-8"))
    content = path.read_bytes()
    assert record["golden"] == name(path)
    assert (record["sha256"], record["bytes"]) == (hashlib.sha256(content).hexdigest(), len(content))
    assert record["spec"] == "007-FR-3"
    assert record["oracle"]["path"] == ORACLES[path.relative_to(GOLDEN).parts[0]]
    assert SHA256.fullmatch(record["oracle"]["sha256"])
    assert record["capture_script"]["path"] == "scripts/capture_golden_007.py"
    assert SHA256.fullmatch(record["capture_script"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", record["repo_commit"])
    assert date.fromisoformat(record["captured"])
    assert record["fixture"]["builder"] and isinstance(record["fixture"]["seeds"], dict)
    assert [h for h in record["helpers"] if not hasattr(manager_helpers, h)] == []
    assert record["settings"] and record["replaces"]
    assert record["map"]["sha256"] is None or SHA256.fullmatch(record["map"]["sha256"])
    assert all(i["name"] and i["bytes"] > 0 and SHA256.fullmatch(i["sha256"]) for i in record["inputs"])


def test_every_provenance_record_has_its_golden_file():
    assert [name(p) for p in RECORDS if not p.with_name(p.name[:-len(".provenance.json")]).is_file()] == []
    assert len(RECORDS) == len(FILES)


def test_tracked_code_never_imports_reference_scripts():
    offenders = []
    for path in sorted([*REPO.glob("src/**/*.py"), *REPO.glob("exe_programs/**/*.py"), *REPO.glob("tests/*.py")]):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level
                       else [])
            offenders += [f"{path.relative_to(REPO)}:{node.lineno} {m}" for m in modules
                          if m.split(".")[0] in FORBIDDEN]
    assert offenders == []
