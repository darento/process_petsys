"""tests/manager_helpers.py smoke tests (spec 007 FR-6, T3): imports headless, builds each fixture."""

import os
import subprocess
import sys
import unittest

import pytest

from helpers import REPO
from manager_helpers import (CLIFixtures, CalibrationFixtures, DirectDummyChild, FakeBackend, FakeChild,
                             FakeDaemon, FakeResources, FixtureProbe, ListmodeFixtures, P3_SPECS, PrivateOutput,
                             QCFixtures, ToolWorld, compact_record, encode_compact, fixed_record, pairs,
                             profile_fixture, sides_for)
from src.cornell.inputs import validate_ldat
from src.petsys_manager.acquisition import DaqdConfig
from src.petsys_manager.contracts import Action, CommandSpec, DataFormat, Identity, InputDescriptor
from src.petsys_manager.settings import preflight
from src.read_compact import read_binary_file


@pytest.mark.fr("007-FR-6")
def test_imports_without_tk_or_display():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "tests")])
    code = "import sys, manager_helpers; sys.exit('tkinter' in sys.modules or 'customtkinter' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.fr("007-FR-6")
def test_profile_fixture_passes_preflight(tmp_path):
    root = tmp_path / "profile"
    report = preflight(profile_fixture(root), Action.LISTMODE, None,
                       (InputDescriptor("private/input.ldat", "compact", "coincidence"),),
                       repo_root=root, probe=FixtureProbe())
    assert report.ready, report.issues


@pytest.mark.fr("007-FR-6")
def test_wire_encoders_round_trip(tmp_path):
    sides = ([(10, 2.5, 0), (11, 3.0, 128)], [(20, 4.0, 262144)])
    assert len(fixed_record(sides)) == 2 + 2 * 4 * 16
    path = tmp_path / "pair.ldat"
    path.write_bytes(compact_record(sides))
    other = tmp_path / "pair_encoded.ldat"
    encode_compact(other, [sides])
    assert other.read_bytes() == path.read_bytes()
    (det1, det2), = read_binary_file(str(path))
    assert [h[2] for h in det1] == [0, 128] and [h[2] for h in det2] == [262144]


@pytest.mark.fr("007-FR-6")
def test_calibration_fixtures_write_valid_ldat(tmp_path):
    fx = CalibrationFixtures(tmp_path / "cal")
    records = pairs(sides_for(fx.geometry, P3_SPECS[:2], seed=5))[:200]
    for fmt in (DataFormat.COMPACT, DataFormat.FIXED):
        descriptor = fx.write(f"cal_{fmt.value}.ldat", records, fmt)
        assert descriptor.path.stat().st_size > 0
        summary = validate_ldat(descriptor, fx.mapping.modules)
        assert summary.records == len(records)
    assert fx.limits().path.is_file()


@pytest.mark.fr("007-FR-6")
def test_listmode_fixtures_standalone(tmp_path):
    h = ListmodeFixtures.at(tmp_path / "lm")
    descriptors, maps = h.inputs(count=60, random_slabs=False)
    assert [d.path.name for d in descriptors] == ["acq_coinc_2.ldat", "acq_coinc_10.ldat"]
    assert set(maps) == {"calibration", "cog_limits", "doi_limits", "pairs", "regions", "metadata"}
    twins = h.compact_twins(descriptors, count=60, random_slabs=False)
    assert all(t.path.is_file() for t in twins)


@pytest.mark.fr("007-FR-6")
def test_qc_fixtures_standalone(tmp_path):
    h = QCFixtures.at(tmp_path / "qc")
    descriptors = h.inputs(count=60)
    assert len(list(read_binary_file(str(descriptors[0].path)))) == 60
    assert h.results.is_dir() and len(h.manual_records()) == 5


@pytest.mark.fr("007-FR-6")
def test_process_fakes():
    backend = FakeBackend()
    assert backend.launch("cmd") is backend.child and backend.launched == ["cmd"]
    child = FakeChild(None)
    child.terminate()
    assert child.signals == ["TERM"] and child.poll() == -15
    daemon = FakeDaemon()
    resources = FakeResources(serve_after=1)
    resources.daemon = daemon
    config = DaqdConfig(("daqd",), "/tmp", "/tmp/d.sock", "/dev/shm/d")
    resources.appear(config)
    assert resources.existing(config) == ("/tmp/d.sock", "/dev/shm/d")
    with pytest.raises(OSError):
        resources.query("/tmp/d.sock", 0.1, None)
    assert resources.query("/tmp/d.sock", 0.1, None) == ("/daqd_shm", 4242)
    daemon.die(0)
    with pytest.raises(OSError, match="refused"):
        resources.query("/tmp/d.sock", 0.1, None)
    direct = DirectDummyChild(subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE))
    assert direct.wait(30) == 0
    direct.stdout.close()
    direct.stderr.close()


@pytest.mark.fr("007-FR-6")
def test_tool_world_fakes_acquisition(tmp_path):
    class Check:
        root = tmp_path
    world = ToolWorld(Check())
    prefix = tmp_path / "acq"
    command = CommandSpec(("acquire_sipm_data", "-o", str(prefix)), tmp_path,
                          identity=Identity("run", "acquire", "attempt"))
    child = world.launch(command)
    assert child.poll() == 0
    assert (tmp_path / "acq.rawf").stat().st_size == 64 and (tmp_path / "acq.idxf").is_file()


class ListmodeMixinSmoke(ListmodeFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-t3-lm-"

    @pytest.mark.fr("007-FR-6")
    def test_mixin_roots_at_test_name(self):
        self.assertEqual(self.root, self.output / self._testMethodName)
        self.assertTrue(self.map_path.is_file())


class CLIMixinSmoke(CLIFixtures, unittest.TestCase):
    @pytest.mark.fr("007-FR-6")
    def test_cli_qc_request_paths(self):
        self.assertTrue(self.settings.profile)
        h, descriptors, request = self.qc_request(count=40)
        self.assertEqual(request["action"], "qc")
        self.assertTrue(all(d.path.is_file() for d in descriptors))
        path, result = self.write_request(request)
        self.assertTrue(path.is_file() and not result.exists())
        self.assertIn(" ; & ", str(path))
