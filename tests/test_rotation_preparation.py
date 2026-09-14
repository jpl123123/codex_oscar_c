from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oscar_ascend.calibration_data import build_prompts, digest
from oscar_ascend.plugin import PluginConfig
from oscar_ascend.prepare_rotations import ensure_rotations

ROOT = Path(__file__).resolve().parents[1]
CSV = b'Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3\nQ?,correct,wrong1,wrong2,wrong3\n'


class RotationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"text_config":{"head_dim":256}}')
        self.library = self.root / "ops.so"
        self.library.write_bytes(b"mock library identity; never loaded")
        target = json.loads((ROOT / "configs/target_service.json").read_text())
        target["argv"][2] = str(self.model)
        self.target = self.root / "service.json"
        self.target.write_text(json.dumps(target))
        self.profile = self.root / "profile.json"
        profile = json.loads((ROOT / "configs/calibration.json").read_text())
        profile.update(num_prompts=1, dataset_sha256=hashlib.sha256(CSV).hexdigest())
        self.profile.write_text(json.dumps(profile))
        self.calls = []

    def generate(self, request_path, candidate, profile):
        request = json.loads(request_path.read_text())
        self.assertEqual(request["token_budget"], 30000)
        self.assertEqual(request["tensor_parallel_size"], 4)
        self.assertEqual(request["prompts"], build_prompts(CSV, 1, 0))
        self.calls.append(request)
        candidate.write_text(json.dumps({"key": request["calibration_request_sha256"]}))

    def validate(self, path, identity):
        if json.loads(path.read_text()) != {"key": digest(identity)}:
            raise ValueError("mock artifact identity mismatch")

    def run_preparation(self, options=None, generate=None):
        return ensure_rotations(service_config=self.target, options=options or PluginConfig(),
                                library=self.library, profile_path=self.profile,
                                cache_root=self.root / "artifacts/rotations", generate=generate or self.generate,
                                validator=self.validate, corpus_loader=lambda *_: CSV)

    def test_missing_generates_once_then_reuses_without_engine(self):
        first = self.run_preparation()
        self.assertEqual(first["status"], "generated")
        second = self.run_preparation()
        self.assertEqual(second["status"], "reused")
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(len(self.calls), 1)

    def test_model_change_selects_another_artifact(self):
        first = self.run_preparation()
        (self.model / "config.json").write_text('{"text_config":{"head_dim":128}}')
        second = self.run_preparation()
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(len(self.calls), 2)

    def test_tokenizer_or_chat_template_change_invalidates_rotation(self):
        first = self.run_preparation()
        (self.model / "chat_template.jinja").write_text("new template {{ messages }}")
        second = self.run_preparation()
        self.assertNotEqual(first["path"], second["path"])

    def test_missing_manifest_does_not_reuse_an_unpublished_pt(self):
        first = self.run_preparation()
        Path(first["path"]).with_name("manifest.json").unlink()
        second = self.run_preparation()
        self.assertEqual(second["status"], "generated")
        self.assertEqual(len(self.calls), 2)

    def test_invalid_auto_artifact_is_preserved_then_regenerated(self):
        first = self.run_preparation()
        Path(first["path"]).write_text("corrupt")
        second = self.run_preparation()
        self.assertEqual(second["status"], "generated")
        self.assertEqual(len(list(Path(first["path"]).parent.glob("invalid-*.pt"))), 1)

    def test_invalid_explicit_artifact_is_never_overwritten(self):
        explicit = self.root / "user.pt"
        explicit.write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "preserved"):
            self.run_preparation(replace(PluginConfig(), rotations_path=str(explicit)))
        self.assertEqual(explicit.read_text(), "corrupt")
        self.assertEqual(self.calls, [])

    def test_failed_generation_does_not_publish_partial_pt(self):
        def fail(request_path, candidate, profile):
            candidate.write_text("half-written")
            raise RuntimeError("NPU worker failed")
        with self.assertRaisesRegex(RuntimeError, "NPU worker failed"):
            self.run_preparation(generate=fail)
        self.assertFalse(list((self.root / "artifacts/rotations").glob("*/rotations.pt")))

    def test_invalid_generator_output_never_publishes(self):
        def wrong(request_path, candidate, profile):
            candidate.write_text('{"key":"wrong"}')
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_preparation(generate=wrong)
        self.assertFalse(list((self.root / "artifacts/rotations").glob("*/rotations.pt")))


if __name__ == "__main__":
    unittest.main()
