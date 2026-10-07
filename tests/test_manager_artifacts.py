"""RunStore reservations, manifests and artifact ownership (spec 003 T4, T29, T34; spec 007 T4).

Moved from scripts/petsys_manager_artifact_check.py; temporary files only.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest

from manager_helpers import PrivateOutput
from src.petsys_manager.artifacts import ArtifactError, RunStore, read_manifest
from src.petsys_manager.contracts import (Artifact, CommandResult, Identity,
    InputDescriptor, ResultStatus)


@pytest.mark.fr("003-FR-9", "003-FR-11", "003-FR-16")  # spec 003 T4
class ArtifactChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-artifacts-"

    def setUp(self):
        self.root = self.output / self._testMethodName
        self.root.mkdir()
        self.settings = {"calibration": "operator supplied", "cuts": {"keV": [357, 665]}}
        self.store = RunStore.reserve(self.root, self.settings, run_id="fixture")
        self.attempt = self.store.reserve_attempt("convert", attempt_id="attempt-1")

    def file(self, name, content=b"", *, directory=None):
        path = (directory or self.attempt.directory) / name
        with path.open("xb") as out:
            out.write(content)
        return path

    def artifact(self, name="output.ldat", content=b"fixture", *, disposable=False, validated=False):
        path = self.file(name, content)
        return Artifact(path, "ldat", InputDescriptor(path, "fixed", "coincidence", validated), disposable)

    def result(self, status="succeeded", artifacts=(), *, attempt=None):
        return CommandResult((attempt or self.attempt).identity, status,
            0 if status == "succeeded" else 7 if status == "failed" else None,
            "Fixture outcome", artifacts, outputs_validated=status == "succeeded")

    def test_run_collision_preserves_existing_and_never_adopts_empty_directory(self):
        before = self.store.manifest_path.read_bytes()
        with self.assertRaises(FileExistsError):
            RunStore.reserve(self.root, {}, run_id="fixture")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            RunStore.reserve(self.root, {}, run_id="empty")
        self.assertEqual(self.store.manifest_path.read_bytes(), before)
        self.assertEqual(tuple(empty.iterdir()), ())

    def test_attempt_collision_and_retry_destinations_preserve_first_attempt(self):
        old = self.file("failed.rawf", b"raw")
        with self.assertRaises(ArtifactError):
            self.store.reserve_attempt("convert", attempt_id="attempt-2")
        self.store.finish_attempt(self.attempt, self.result("failed", [Artifact(old, "raw")]))
        with self.assertRaises(FileExistsError):
            self.store.reserve_attempt("convert", attempt_id="attempt-1")
        retry = self.store.reserve_attempt("convert", attempt_id="attempt-2")
        self.assertNotEqual(retry.directory, self.attempt.directory)
        self.assertEqual(old.read_bytes(), b"raw")
        self.assertEqual([a["status"] for a in read_manifest(self.store.root)["attempts"]],
                         ["failed", "partial"])

    def test_preexisting_stage_directory_is_not_adopted(self):
        directory = self.store.root / "calibrate"
        directory.mkdir()
        self.file("private", b"keep", directory=directory)
        with self.assertRaises(FileExistsError):
            self.store.reserve_attempt("calibrate")
        self.assertEqual((directory / "private").read_bytes(), b"keep")

    def test_generated_run_and_attempt_ids_are_distinct(self):
        other = RunStore.reserve(self.root, {})
        self.assertNotEqual(self.store.root, other.root)
        one = other.reserve_attempt("acquire")
        other.finish_attempt(one, self.result("cancelled", attempt=one))
        two = other.reserve_attempt("acquire")
        self.assertNotEqual(one.identity, two.identity)

    def test_settings_and_exact_ordered_inputs_are_snapshotted(self):
        first = self.file("data_10.ldat", b"123")
        second = self.file("data_2.ldat", b"12")
        inputs = [InputDescriptor(first, "compact", "coincidence"),
                  InputDescriptor(second, "fixed", "group")]
        store = RunStore.reserve(self.root, self.settings, inputs, run_id="inputs")
        self.settings["cuts"]["keV"][0] = -1
        inputs.reverse()
        snapshot = read_manifest(store.root)
        self.assertEqual(snapshot["settings"]["cuts"]["keV"], (357, 665))
        self.assertEqual([i["path"] for i in snapshot["inputs"]], [str(first), str(second)])
        self.assertEqual([i["size_bytes"] for i in snapshot["inputs"]], [3, 2])
        self.assertEqual([i["population"] for i in snapshot["inputs"]], ["coincidence", "group"])
        with self.assertRaises(TypeError):
            store.snapshot["status"] = "succeeded"
        self.assertEqual(first.read_bytes(), b"123")

    def test_invalid_settings_and_inputs_fail_before_run_reservation(self):
        for value in ({"nan": float("nan")}, {"object": object()}, []):
            with self.subTest(value=str(value)), self.assertRaises((ValueError, TypeError)):
                RunStore.reserve(self.root, value, run_id="bad")
            self.assertFalse((self.root / "bad").exists())
        missing = InputDescriptor(self.root / "missing", "fixed", "coincidence")
        with self.assertRaises(FileNotFoundError):
            RunStore.reserve(self.root, {}, [missing], run_id="bad")

    def test_duplicate_inputs_and_untyped_inputs_rejected(self):
        item = self.artifact().input_descriptor
        for inputs in ([item, item], [item.path]):
            with self.assertRaises(ArtifactError):
                RunStore.reserve(self.root, {}, inputs, run_id="bad")

    def test_traversal_absolute_identity_and_device_names_rejected(self):
        for name in ("..", "a/b", "a\\b", "a:stream", "CON", "COM1", "", "/tmp", "a\x00b"):
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                self.store.reserve_attempt(name)
        for name in ("..", "C:\\outside", "NUL"):
            with self.assertRaises(ArtifactError):
                RunStore.reserve(self.root, {}, run_id=name)
        with self.assertRaises(ArtifactError):
            RunStore.reserve(self.root / ".." / "escape", {})

    def test_foreign_and_forged_attempts_are_refused(self):
        from src.petsys_manager.artifacts import Attempt
        foreign = replace(self.attempt, identity=Identity("other", "convert", "attempt-1"))
        forged = Attempt(self.attempt.identity, self.root)
        for attempt in (foreign, forged):
            with self.assertRaises(ArtifactError):
                self.store.inventory(attempt)
        with self.assertRaises(ArtifactError):
            self.store.finish_attempt(self.attempt, self.result("failed", attempt=foreign))

    def test_exact_converter_outputs_natural_order_without_first_split(self):
        prefix = self.attempt.directory / "literal & [data]"
        before = self.store.inventory(self.attempt)
        for name in ("literal & [data]_00000010.ldat", "literal & [data]_00000002.ldat",
                     "literal & [data]_00000100.ldat", "literal & [data]_other.ldat",
                     "literal & [data]2.ldat", "literal & [data]_00000002.lidx", "other.ldat"):
            self.file(name, b"keep")
        artifacts = self.store.discover_ldat(self.attempt, prefix, "compact", "coincidence", before=before)
        self.assertEqual([a.path.name for a in artifacts],
                         ["literal & [data]_00000002.ldat", "literal & [data]_00000010.ldat",
                          "literal & [data]_00000100.ldat"])
        self.assertTrue(all(not a.disposable and not a.input_descriptor.validated for a in artifacts))
        self.store.record_artifacts(self.attempt, artifacts)
        self.assertEqual(len(tuple(self.attempt.directory.iterdir())), 7)
        self.assertEqual(len(read_manifest(self.store.root)["attempts"][0]["artifacts"]), 3)

    def test_unsplit_converter_output_and_empty_inventory_are_exact(self):
        prefix = self.attempt.directory / "output"
        self.assertEqual(self.store.discover_ldat(self.attempt, prefix, "fixed", "group", before=()), ())
        path = self.file("output.ldat", b"fixed-group")
        artifacts = self.store.discover_ldat(self.attempt, prefix, "fixed", "group", before=())
        self.assertEqual(tuple(a.path for a in artifacts), (path,))
        self.assertEqual(artifacts[0].input_descriptor.population.value, "group")

    def test_stale_matching_converter_file_is_collision(self):
        old = self.file("output_00000004.ldat", b"old")
        before = self.store.inventory(self.attempt)
        self.file("output_00000009.ldat", b"new")
        with self.assertRaises(ArtifactError):
            self.store.discover_ldat(self.attempt, self.attempt.directory / "output",
                                     "fixed", "coincidence", before=before)
        self.assertEqual(old.read_bytes(), b"old")

    def test_artifact_metadata_wrong_path_or_duplicate_is_rejected(self):
        artifact = self.artifact()
        wrong = replace(artifact, input_descriptor=replace(artifact.input_descriptor,
                                                          path=self.root / "different"))
        for artifacts in ([wrong], [artifact, artifact], [Artifact(self.root / "outside", "raw")]):
            with self.assertRaises((ArtifactError, FileNotFoundError)):
                self.store.record_artifacts(self.attempt, artifacts)
        self.store.record_artifacts(self.attempt, [artifact])
        with self.assertRaises(ArtifactError):
            self.store.record_artifacts(self.attempt, [artifact])

    def test_recorded_format_population_and_ownership_cannot_change_at_completion(self):
        artifact = self.artifact()
        self.store.record_artifacts(self.attempt, [artifact])
        valid = replace(artifact, input_descriptor=replace(artifact.input_descriptor, validated=True))
        for changed in (replace(valid, kind="raw"),
                        replace(valid, input_descriptor=replace(valid.input_descriptor, format="compact")),
                        replace(valid, input_descriptor=replace(valid.input_descriptor, population="group"))):
            with self.assertRaises(ArtifactError):
                self.store.finish_attempt(self.attempt, self.result(artifacts=[changed]))
        self.store.finish_attempt(self.attempt, self.result(artifacts=[valid]))
        self.assertEqual(read_manifest(self.store.root)["attempts"][0]["status"], "succeeded")

    def test_ldat_without_format_descriptor_and_changed_run_root_are_refused(self):
        path = self.file("ambiguous.ldat", b"data")
        with self.assertRaises(ArtifactError):
            self.store.record_artifacts(self.attempt, [Artifact(path, "ldat")])
        with self.assertRaises(AttributeError):
            self.store.root = self.root
        with self.assertRaises(AttributeError):
            self.store.run_id = "other"

    def test_cleanup_only_recorded_empty_disposable_ldat(self):
        artifact = self.artifact(content=b"", disposable=True)
        self.store.record_artifacts(self.attempt, [artifact])
        self.store.remove_disposable(self.attempt, artifact.path)
        self.assertFalse(artifact.path.exists())
        self.assertEqual(read_manifest(self.store.root)["attempts"][0]["artifacts"][0]["cleanup"], "removed")

    @pytest.mark.fr("003-FR-9")
    def test_converter_index_discovered_exactly_and_removable(self):
        """T34: `<prefix>(_N).lidx` new since the inventory, disposable; a look-alike or changed index is kept."""
        prefix = self.attempt.directory / "acq_coincCompact"
        old = self.file("acq_coincCompact_3.lidx", b"old")
        before = self.store.inventory(self.attempt)
        old.unlink()
        names = ("acq_coincCompact_00000000.lidx", "acq_coincCompact_00000010.lidx", "acq_coincCompact_2.lidx",
                 "acq_coincCompact.lidx", "acq_coincCompact_2.ldat", "acq_coincCompact_x.lidx",
                 "acq_coincCompact2_1.lidx", "other_coincCompact_1.lidx", "acq_coincCompact_1.lidx.bak")
        for name in names:
            self.file(name, b"idx")
        found = self.store.discover_index(self.attempt, prefix, before=before)
        self.assertEqual([a.path.name for a in found], ["acq_coincCompact.lidx", "acq_coincCompact_00000000.lidx",
                                                        "acq_coincCompact_2.lidx", "acq_coincCompact_00000010.lidx"])
        self.assertTrue(all(a.kind == "index" and a.disposable for a in found))
        self.file("acq_coincCompact_3.lidx", b"old")
        with self.assertRaises(ArtifactError):          # pre-existing look-alike: a collision, never adopted
            self.store.discover_index(self.attempt, prefix, before=before)
        (self.attempt.directory / "acq_coincCompact_3.lidx").unlink()
        self.store.record_artifacts(self.attempt, found)
        found[-1].path.write_bytes(b"changed")
        for artifact in found[:-1]:
            self.store.remove_disposable(self.attempt, artifact.path)
            self.assertFalse(artifact.path.exists())
        with self.assertRaises(ArtifactError):
            self.store.remove_disposable(self.attempt, found[-1].path)
        with self.assertRaises(ArtifactError):          # not recorded
            self.store.remove_disposable(self.attempt, self.attempt.directory / "acq_coincCompact_x.lidx")
        with self.assertRaises(ArtifactError):          # an index can be disposable only as a .lidx
            self.store.record_artifacts(self.attempt, [Artifact(self.file("index.idx", b"i"), "index",
                                                                disposable=True)])
        cleanup = {Path(a["path"]).name: a["cleanup"]
                   for a in read_manifest(self.store.root)["attempts"][0]["artifacts"]}
        self.assertEqual(cleanup, {"acq_coincCompact.lidx": "removed", "acq_coincCompact_00000000.lidx": "removed",
                                   "acq_coincCompact_2.lidx": "removed", "acq_coincCompact_00000010.lidx": "retained"})
        self.assertEqual(sorted(p.name for p in self.attempt.directory.iterdir() if p.suffix == ".lidx"),
                         ["acq_coincCompact2_1.lidx", "acq_coincCompact_00000010.lidx", "acq_coincCompact_x.lidx",
                          "other_coincCompact_1.lidx"])

    def test_failed_raw_nonempty_ldat_index_and_unrelated_files_survive(self):
        artifacts = [Artifact(self.file("output.rawf", b"raw"), "raw"),
                     self.artifact(), Artifact(self.file("output.lidx", b""), "index"),
                     self.artifact("unmarked.ldat", b""),
                     Artifact(self.file("unrelated.rawf", b"private"), "raw")]
        self.store.record_artifacts(self.attempt, artifacts)
        self.store.finish_attempt(self.attempt, self.result("failed"))
        before = {a.path: a.path.read_bytes() for a in artifacts}
        for artifact in artifacts:
            with self.assertRaises(ArtifactError):
                self.store.remove_disposable(self.attempt, artifact.path)
        self.store.finish("failed", "conversion failed")
        self.assertEqual({a.path: a.path.read_bytes() for a in artifacts}, before)
        self.assertEqual(read_manifest(self.store.root)["status"], "failed")

    def test_unrecorded_empty_ldat_cannot_be_removed(self):
        path = self.file("unrecorded.ldat")
        with self.assertRaises(ArtifactError):
            self.store.remove_disposable(self.attempt, path)
        self.assertTrue(path.exists())

    def test_nonempty_or_non_ldat_cannot_be_marked_disposable(self):
        for artifact in (self.artifact(content=b"data", disposable=True),
                         Artifact(self.file("empty.rawf"), "raw", disposable=True),
                         Artifact(self.file("empty.lidx"), "ldat", disposable=True)):
            with self.assertRaises(ArtifactError):
                self.store.record_artifacts(self.attempt, [artifact])
            self.assertTrue(artifact.path.exists())

    def test_grown_or_replaced_empty_file_cannot_be_removed(self):
        artifact = self.artifact(content=b"", disposable=True)
        self.store.record_artifacts(self.attempt, [artifact])
        artifact.path.write_bytes(b"new data")
        with self.assertRaises(ArtifactError):
            self.store.remove_disposable(self.attempt, artifact.path)
        self.assertEqual(artifact.path.read_bytes(), b"new data")
        artifact.path.rename(artifact.path.with_suffix(".kept"))
        artifact.path.touch(exist_ok=False)
        with self.assertRaises(ArtifactError):
            self.store.remove_disposable(self.attempt, artifact.path)
        self.assertTrue(artifact.path.exists())

    def test_cleanup_intent_persisted_if_unlink_fails(self):
        artifact = self.artifact(content=b"", disposable=True)
        self.store.record_artifacts(self.attempt, [artifact])
        target = "pathlib.Path.unlink" if os.name == "nt" else "src.petsys_manager.artifacts.os.unlink"
        original = Path.unlink if os.name == "nt" else os.unlink
        def fail(path, *args, **kwargs):
            if Path(path).name == artifact.path.name:
                raise OSError("fixture unlink failed")
            return original(path, *args, **kwargs)
        with patch(target, side_effect=fail, autospec=True), self.assertRaises(OSError):
            self.store.remove_disposable(self.attempt, artifact.path)
        manifest = read_manifest(self.store.root)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["attempts"][0]["artifacts"][0]["cleanup"], "pending")
        self.assertTrue(artifact.path.exists())

    def test_publish_exclusive_and_preserve_partial_on_collision(self):
        partial = self.file("output.ldat.partial", b"generated")
        existing = self.file("output.ldat", b"existing")
        with self.assertRaises(FileExistsError):
            self.store.publish(self.attempt, partial, existing)
        self.assertEqual(existing.read_bytes(), b"existing")
        self.assertEqual(partial.read_bytes(), b"generated")
        self.assertEqual(read_manifest(self.store.root)["status"], "partial")

    def test_publish_success_is_not_workflow_success(self):
        partial = self.file("output.ldat.partial", b"generated")
        target = self.attempt.directory / "output.ldat"
        self.assertEqual(self.store.publish(self.attempt, partial, target), target)
        self.assertEqual(target.read_bytes(), b"generated")
        self.assertEqual(read_manifest(self.store.root)["status"], "partial")
        self.assertTrue(partial.exists())

    def test_publication_failure_and_interruption_preserve_partial(self):
        partial = self.file("output.ldat.partial", b"generated")
        for failure in (OSError("link unavailable"), KeyboardInterrupt()):
            with patch("src.petsys_manager.artifacts._link", side_effect=failure):
                with self.assertRaises(type(failure)):
                    self.store.publish(self.attempt, partial, self.attempt.directory / "output.ldat")
            self.assertEqual(read_manifest(self.store.root)["status"], "partial")
            self.assertTrue(partial.exists())

    def test_manifest_failure_or_interruption_cannot_record_final_success(self):
        artifact = self.artifact(validated=True)
        self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        old = self.store.manifest_path.read_bytes()
        for failure in (OSError("disk full"), KeyboardInterrupt()):
            with patch("src.petsys_manager.artifacts._link", side_effect=failure):
                with self.assertRaises(type(failure)):
                    self.store.finish("succeeded")
            self.assertEqual(read_manifest(self.store.root)["status"], "partial")
            self.assertEqual(self.store.manifest_path.read_bytes(), old)
            self.assertEqual(self.store.snapshot["status"], "partial")
        self.assertTrue(any(path.suffix == ".partial" for path in (self.store.root / ".history").iterdir()))

    def test_output_flush_failure_cannot_complete_attempt(self):
        artifact = self.artifact(validated=True)
        with patch("src.petsys_manager.artifacts._sync_file", side_effect=OSError("flush failed")):
            with self.assertRaises(OSError):
                self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        self.assertEqual(read_manifest(self.store.root)["attempts"][0]["status"], "partial")
        self.assertEqual(artifact.path.read_bytes(), b"fixture")

    def test_manifest_directory_sync_failure_rolls_back_new_success_revision(self):
        artifact = self.artifact(validated=True)
        self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        with patch("src.petsys_manager.artifacts._sync_directory", side_effect=OSError("sync failed")):
            with self.assertRaises(OSError):
                self.store.finish("succeeded")
        self.assertEqual(read_manifest(self.store.root)["status"], "partial")
        self.assertEqual(self.store.snapshot["status"], "partial")

    def test_manifest_revision_collision_never_overwrites(self):
        collision = self.store.root / ".history/manifest-000003.json"
        collision.write_bytes(b"private metadata")
        with self.assertRaises(FileExistsError):
            self.store.record_artifacts(self.attempt, [self.artifact()])
        self.assertEqual(collision.read_bytes(), b"private metadata")
        self.assertEqual(self.store.snapshot["status"], "partial")

    @pytest.mark.fr("003-FR-24")
    def test_zero_exit_or_unvalidated_descriptor_never_means_success(self):
        artifact = self.artifact()
        result = replace(self.result(artifacts=[artifact]), outputs_validated=False)
        for result in (result, self.result()):
            with self.assertRaises(ArtifactError):
                self.store.finish_attempt(self.attempt, result)
        self.assertEqual(read_manifest(self.store.root)["attempts"][0]["status"], "partial")
        # FR-24: a structure-checked LDAT (descriptor not validated) may be a successful output, and the
        # manifest records that its descriptor was not validated.
        self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        record = read_manifest(self.store.root)["attempts"][0]
        self.assertEqual(record["status"], "succeeded")
        self.assertFalse(record["artifacts"][0]["input_descriptor"]["validated"])

    def test_success_requires_latest_stage_attempt_and_unchanged_outputs(self):
        with self.assertRaises(ArtifactError):
            self.store.finish("succeeded")
        artifact = self.artifact(validated=True)
        self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        artifact.path.write_bytes(b"changed")
        with self.assertRaises(ArtifactError):
            self.store.finish("succeeded")
        self.assertEqual(read_manifest(self.store.root)["status"], "partial")

    def test_retry_success_preserves_failed_attempt_and_immutable_history(self):
        failed = self.artifact("failed.ldat", b"partial data")
        self.store.finish_attempt(self.attempt, self.result("failed", [failed]))
        old = {path: path.read_bytes() for path in self.store.root.glob(".history/manifest-*.json")}
        retry = self.store.reserve_attempt("convert", attempt_id="attempt-2")
        path = self.file("output_00000002.ldat", b"valid", directory=retry.directory)
        artifact = Artifact(path, "ldat", InputDescriptor(path, "compact", "coincidence", True))
        self.store.finish_attempt(retry, self.result(artifacts=[artifact], attempt=retry))
        self.store.finish("succeeded", "validated fixture")
        manifest = read_manifest(self.store.root)
        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual([a["status"] for a in manifest["attempts"]], ["failed", "succeeded"])
        self.assertEqual(manifest["attempts"][1]["outputs"], (str(path),))
        self.assertEqual(failed.path.read_bytes(), b"partial data")
        self.assertTrue(all(path.read_bytes() == content for path, content in old.items()))
        with self.assertRaises(ArtifactError):
            self.store.reserve_attempt("calibrate")

    def test_failed_cancelled_and_launch_error_are_durable_distinct_states(self):
        for status in ("failed", "cancelled", "launch_error"):
            store = RunStore.reserve(self.root, {}, run_id=status)
            attempt = store.reserve_attempt("acquire")
            store.finish_attempt(attempt, self.result(status, attempt=attempt))
            store.finish(status, "fixture reason")
            self.assertEqual(read_manifest(store.root)["status"], status)
            self.assertEqual(read_manifest(store.root)["attempts"][0]["status"], status)

    def test_reader_rejects_corrupted_duplicate_nonfinite_and_oversized_latest_revision(self):
        newer = self.store.root / ".history/manifest-000003.json"
        for content in (b'{"schema_version": 1, "schema_version": 1}',
                        b'{"status": NaN}', b'[]', b'not json'):
            newer.write_bytes(content)
            with self.assertRaises((ArtifactError, ValueError)):
                read_manifest(self.store.root)
        newer.write_bytes(b"x" * 65)
        with patch.object(RunStore, "manifest_limit_bytes", 64), self.assertRaises(ArtifactError):
            read_manifest(self.store.root)

    def test_inventory_and_manifest_metadata_bounds_fail_closed(self):
        self.file("one.ldat")
        self.file("two.ldat")
        with patch.object(RunStore, "artifact_limit", 1):
            with self.assertRaises(ArtifactError):
                self.store.inventory(self.attempt)
            with self.assertRaises(ArtifactError):
                self.store.record_artifacts(self.attempt,
                    [Artifact(self.attempt.directory / name, "ldat",
                              InputDescriptor(self.attempt.directory / name, "fixed", "coincidence"))
                     for name in ("one.ldat", "two.ldat")])
        with patch.object(RunStore, "manifest_limit_bytes", 64), self.assertRaises(ArtifactError):
            path = self.attempt.directory / "one.ldat"
            self.store.record_artifacts(self.attempt,
                [Artifact(path, "ldat", InputDescriptor(path, "fixed", "coincidence"))])
        self.assertEqual(read_manifest(self.store.root)["status"], "partial")

    @pytest.mark.linux  # symlink creation needs privileges on Windows
    def test_symlink_destination_attempt_output_and_manifest_escape_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        private = self.file("private.ldat", b"private", directory=outside)
        linked_destination = self.root / "linked"
        linked_destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactError):
            RunStore.reserve(linked_destination, {})
        linked_output = self.attempt.directory / "output.ldat"
        linked_output.symlink_to(private)
        with self.assertRaises(ArtifactError):
            self.store.record_artifacts(self.attempt, [Artifact(linked_output, "ldat")])
        with self.assertRaises(ArtifactError):
            self.store.discover_ldat(self.attempt, self.attempt.directory / "output", "fixed",
                                     "coincidence", before=())
        manifest_link = self.store.root / ".history/manifest-000003.json"
        manifest_link.symlink_to(private)
        with self.assertRaises(ArtifactError):
            read_manifest(self.store.root)
        self.assertEqual(private.read_bytes(), b"private")
        self.attempt.directory.rename(self.attempt.directory.with_name("saved"))
        self.attempt.directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactError):
            self.store.inventory(self.attempt)
        self.assertEqual(private.read_bytes(), b"private")

    def test_hardlinked_empty_ldat_is_not_disposable_and_external_partial_not_publishable(self):
        artifact = self.artifact(content=b"", disposable=True)
        self.store.record_artifacts(self.attempt, [artifact])
        alias = self.root / "private-link.ldat"
        os.link(artifact.path, alias)
        with self.assertRaises(ArtifactError):
            self.store.remove_disposable(self.attempt, artifact.path)
        partial = self.file("file.partial", b"content")
        os.link(partial, self.root / "external-link")
        with self.assertRaises(ArtifactError):
            self.store.publish(self.attempt, partial, self.attempt.directory / "final")
        self.assertTrue(alias.exists())

    # T29 (FR-9/FR-13): readable names, flat layout, run.json + .history -------------------------

    @pytest.mark.fr("003-FR-13")
    def test_named_run_takes_next_free_suffix_and_never_adopts(self):
        name = "run_0024_lm-P5_2026-10-05_0936"
        roots = [RunStore.reserve(self.root, {}, name=name).root for _ in range(3)]
        self.assertEqual([r.name for r in roots], [name, f"{name}_2", f"{name}_3"])
        self.assertEqual([read_manifest(r)["run_id"] for r in roots], [r.name for r in roots])
        taken = self.root / f"{name}_4"
        taken.mkdir()
        self.assertEqual(RunStore.reserve(self.root, {}, name=name).root.name, f"{name}_5")
        self.assertEqual(tuple(taken.iterdir()), ())
        with patch("src.petsys_manager.artifacts.MAX_NAME_SUFFIX", 5), self.assertRaises(ArtifactError):
            RunStore.reserve(self.root, {}, name=name)
        for options in ({"name": name, "run_id": "x"}, {"name": "a/b"}, {"name": "x", "stages": ()},
                        {"name": "x", "stages": ("lm", "lm")}):
            with self.subTest(options=options), self.assertRaises(ArtifactError):
                RunStore.reserve(self.root, {}, **options)
        self.assertFalse((self.root / "x").exists())

    @pytest.mark.fr("003-FR-13")
    def test_single_stage_writes_in_run_folder_with_one_attempt(self):
        store = RunStore.reserve(self.root, {}, name="data_lm", stages=("listmode",))
        attempt = store.reserve_attempt("listmode", attempt_id="attempt-1")
        self.assertEqual(attempt.directory, store.root)
        for stage in ("listmode", "conversion"):
            with self.subTest(stage=stage), self.assertRaises(ArtifactError):
                store.reserve_attempt(stage)
        output = self.file("data_all.lm", b"lm", directory=store.root)
        self.assertEqual(store.inventory(attempt), (output,))      # run.json and .history are not outputs
        for name in ("run.json", ".history/manifest-000001.json"):
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                store.record_artifacts(attempt, [Artifact(store.root / name, "partial_output")])
        store.finish_attempt(attempt, self.result("failed", [Artifact(output, "partial_output")], attempt=attempt))
        with self.assertRaises(ArtifactError):      # no retry folder: a second attempt would share the folder
            store.reserve_attempt("listmode", attempt_id="attempt-2")
        store.finish("failed", "fixture")
        self.assertEqual(sorted(p.name for p in store.root.iterdir()), [".history", "data_all.lm", "run.json"])
        manifest = read_manifest(store.root)
        self.assertEqual((manifest["stages"], manifest["attempts"][0]["directory"]), (("listmode",), str(store.root)))

    @pytest.mark.fr("003-FR-13")
    def test_multi_stage_numbered_folders_and_acquisition_attempt_folders(self):
        stages = ("acquisition", "conversion", "calibration", "listmode")
        store = RunStore.reserve(self.root, {}, name="F18_pipeline-P5", stages=stages)
        first = store.reserve_attempt("acquisition", attempt_id="attempt-1")
        store.finish_attempt(first, self.result("failed", attempt=first))
        second = store.reserve_attempt("acquisition", attempt_id="attempt-2")
        self.assertEqual([first.directory, second.directory],
                         [store.root / "1_acquisition/attempt-1", store.root / "1_acquisition/attempt-2"])
        store.finish_attempt(second, self.result("cancelled", attempt=second))
        for index, stage in enumerate(stages[1:], 2):
            attempt = store.reserve_attempt(stage, attempt_id="attempt-1")
            self.assertEqual(attempt.directory, store.root / f"{index}_{stage}")
            store.finish_attempt(attempt, self.result("cancelled", attempt=attempt))
        self.assertEqual(sorted(p.name for p in store.root.iterdir()),
                         [".history", "1_acquisition", "2_conversion", "3_calibration", "4_listmode", "run.json"])
        alone = RunStore.reserve(self.root, {}, name="acq", stages=("acquisition",))
        self.assertEqual(alone.reserve_attempt("acquisition", attempt_id="attempt-1").directory,
                         alone.root / "attempt-1")
        clash = RunStore.reserve(self.root, {}, name="clash", stages=("acquisition", "conversion"))
        (clash.root / "2_conversion").mkdir()
        with self.assertRaises(FileExistsError):    # a stage folder is never adopted
            clash.reserve_attempt("conversion")

    @pytest.mark.fr("003-FR-13")
    def test_run_json_is_latest_revision_and_history_is_append_only(self):
        artifact = self.artifact(validated=True)
        self.store.finish_attempt(self.attempt, self.result(artifacts=[artifact]))
        self.store.finish("succeeded", "fixture")
        history = sorted(p.name for p in (self.store.root / ".history").iterdir())
        self.assertEqual(history, [f"manifest-{n:06d}.json" for n in range(1, 5)])
        self.assertEqual(self.store.record_path.read_bytes(), self.store.manifest_path.read_bytes())
        self.assertEqual(json.loads(self.store.record_path.read_text(encoding="utf-8"))["revision"], 4)
        self.assertEqual(read_manifest(self.store.root)["status"], "succeeded")
        self.assertEqual(sorted(p.name for p in self.store.root.iterdir()), [".history", "convert", "run.json"])

    @pytest.mark.fr("003-FR-13")
    def test_failed_run_json_replacement_keeps_previous_record(self):
        old_record, old_revision = self.store.record_path.read_bytes(), self.store.manifest_path
        with patch("src.petsys_manager.artifacts.os.replace", side_effect=OSError("record busy")):
            with self.assertRaises(OSError):
                self.store.record_artifacts(self.attempt, [self.artifact()])
        self.assertEqual(self.store.record_path.read_bytes(), old_record)
        self.assertEqual(self.store.manifest_path, old_revision)
        self.assertFalse((self.store.root / ".history/manifest-000003.json").exists())
        self.assertEqual(read_manifest(self.store.root)["revision"], 2)
        self.assertEqual(self.store.snapshot["attempts"][0]["artifacts"], ())
        self.store.record_artifacts(self.attempt, [Artifact(self.attempt.directory / "output.ldat", "ldat",
            InputDescriptor(self.attempt.directory / "output.ldat", "fixed", "coincidence"))])
        self.assertEqual(read_manifest(self.store.root)["revision"], 3)

    @pytest.mark.fr("003-FR-13")
    def test_pre_t29_run_folder_is_still_read(self):
        legacy = self.root / "convert-20261004-093612-1aab7fd8"
        legacy.mkdir()
        for revision in (1, 2):
            (legacy / f"manifest-{revision:06d}.json").write_text(json.dumps(
                {"schema_version": 1, "run_id": legacy.name, "revision": revision,
                 "status": "succeeded" if revision == 2 else "partial", "attempts": []}), encoding="utf-8")
        manifest = read_manifest(legacy)
        self.assertEqual((manifest["revision"], manifest["status"]), (2, "succeeded"))

    @pytest.mark.fr("003-FR-13")
    def test_reserved_directory_replacement_is_refused(self):
        self.attempt.directory.rename(self.attempt.directory.with_name("saved"))
        self.attempt.directory.mkdir()
        with self.assertRaises(ArtifactError):
            self.store.inventory(self.attempt)

