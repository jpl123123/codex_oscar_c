"""Host contracts only; these tests do not establish NPU execution success."""
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from oscar_ascend.backend import RuntimeCapabilities, RuntimeNotReady, require_runtime
from oscar_ascend.metadata import RequestSlots, Stage, WindowTransition, classify_rows
from oscar_ascend.plugin import (
    NATIVE_FULL_BACKEND, OSCAR_FULL_BACKEND, PluginConfig, enabled,
    install_backend_route,
)
from oscar_ascend.ops import ScratchBudget
from oscar_ascend.runtime import (
    OSCAR_REFERENCE_COMMIT, ROTATION_OBJECTIVES, validate_rotation_metadata,
)


class PluginContractTests(unittest.TestCase):
    def test_fresh_import_does_not_initialize_platform(self):
        script = (
            "import sys; import oscar_ascend.plugin, oscar_ascend.backend, oscar_ascend.metadata; "
            "assert not any(x in sys.modules for x in ('torch','torch_npu','vllm',"
            "'vllm_ascend.device.device_op')); "
            "from oscar_ascend.plugin import register; register()"
        )
        environment = dict(os.environ, OSCAR_ASCEND_ENABLED="0")
        subprocess.run([sys.executable, "-c", script], env=environment, check=True)

    def test_route_preserves_classmethod_and_unrelated_backends(self):
        class Native:
            @classmethod
            def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                                     num_heads=None):
                assert cls is Native
                return selected_backend

        original = inspect.getattr_static(Native, "get_attn_backend_cls")
        original_signature = inspect.signature(Native.get_attn_backend_cls)
        handle = install_backend_route(Native)
        self.assertIs(handle, install_backend_route(Native))
        self.assertEqual(inspect.signature(Native.get_attn_backend_cls), original_signature)
        self.assertEqual(Native.get_attn_backend_cls(NATIVE_FULL_BACKEND, object()),
                         OSCAR_FULL_BACKEND)
        self.assertEqual(Native().get_attn_backend_cls("native_gdn", object()), "native_gdn")
        handle.undo()
        self.assertIs(inspect.getattr_static(Native, "get_attn_backend_cls"), original)
        handle.undo()

    def test_undo_does_not_clobber_later_patch(self):
        class Native:
            @classmethod
            def get_attn_backend_cls(cls, selected_backend, attn_selector_config):
                return selected_backend
        handle = install_backend_route(Native)
        Native.get_attn_backend_cls = classmethod(lambda cls, *args: "other")
        with self.assertRaisesRegex(RuntimeError, "another component"):
            handle.undo()

    def test_configuration_and_runtime_fail_closed(self):
        self.assertFalse(enabled({}))
        self.assertTrue(enabled({"OSCAR_ASCEND_ENABLED": "1"}))
        with self.assertRaises(ValueError):
            enabled({"OSCAR_ASCEND_ENABLED": "yes"})
        with self.assertRaises(ValueError):
            PluginConfig(recent=-1)
        self.assertEqual(PluginConfig(k_clip=0, v_clip=0).k_clip, 0)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(clip=invalid), self.assertRaises(ValueError):
                PluginConfig(k_clip=invalid)
        with self.assertRaises(RuntimeNotReady):
            require_runtime(SimpleNamespace())
        runtime = SimpleNamespace(capabilities=RuntimeCapabilities())
        with self.assertRaisesRegex(RuntimeNotReady, "window_commit"):
            require_runtime(SimpleNamespace(_oscar_attention_runtime=runtime))

    def test_rotation_metadata_binds_model_and_objectives(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            config_bytes = b'{"hidden_size": 256}\n'
            Path(directory, "config.json").write_bytes(config_bytes)
            config = SimpleNamespace(model_config=SimpleNamespace(model=directory),
                                     parallel_config=SimpleNamespace(tensor_parallel_size=4))
            artifact = {
                "format": "oscar-ascend-rotations-v1", "model": directory,
                "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "reference_commit": OSCAR_REFERENCE_COMMIT, "tensor_parallel_size": 4,
                "group_size": 256, "objectives": ROTATION_OBJECTIVES,
                "clip_ratios": {"k": 1.0, "v": 1.0},
            }
            validate_rotation_metadata(artifact, config, PluginConfig())
            for field in ("reference_commit", "model_config_sha256", "objectives", "group_size"):
                with self.subTest(field=field), self.assertRaises(RuntimeNotReady):
                    validate_rotation_metadata({k: v for k, v in artifact.items() if k != field},
                                               config, PluginConfig())
            Path(directory, "config.json").write_bytes(config_bytes + b' ')
            with self.assertRaisesRegex(RuntimeNotReady, "model_config_sha256"):
                validate_rotation_metadata(artifact, config, PluginConfig())


class MetadataContractTests(unittest.TestCase):
    def test_long_chunk_reserves_all_old_recent_migrations(self):
        budget = ScratchBudget(16384, 128, 4, 1, 256, 256, 4, 4096)
        self.assertEqual(budget.migration_tokens, 128 * 260)
        self.assertGreater(budget.tensor_bytes, 128 * 260 * 256 * 4)

    def test_real_query_lengths_and_mixed_stages(self):
        stages = classify_rows(
            [0, 1, 5, 9, 11, 12, 12],
            [0, 4, 100, 100, 100, 0],
            [1, 12, 100, 100, 100, 0],
            [0, 0, 3, 1, 0, 0],
        )
        self.assertEqual(stages, (Stage.PREFILL, Stage.CHUNKED_PREFILL,
                                 Stage.VERIFY, Stage.VERIFY, Stage.DECODE, Stage.PAD))

    def test_verify_is_not_inferred_from_query_length(self):
        with self.assertRaisesRegex(ValueError, "lacks prefill or verify"):
            classify_rows([0, 4], [100], [100], [0])
        self.assertEqual(classify_rows([0, 1], [4], [100], [0]),
                         (Stage.CHUNKED_PREFILL,))
        with self.assertRaisesRegex(ValueError, "scheduled drafts plus one"):
            classify_rows([0, 4], [100], [100], [1])

    def test_commit_never_uses_unmaterialized_bonus_kv(self):
        # Previous verify computed positions 320..323. The sampled bonus at 324
        # has no KV yet and must not be included in the commit length.
        transition = WindowTransition(320, 324, 4, 64, 256)
        self.assertEqual(transition.accepted_kv, 4)
        self.assertEqual(transition.newly_historical, (64, 68))
        with self.assertRaisesRegex(ValueError, "materialized"):
            WindowTransition(320, 325, 4, 64, 256)

    def test_rejection_boundaries_and_short_windows(self):
        for accepted in range(5):
            state = WindowTransition(320, 320 + accepted, 4, 64, 256)
            self.assertEqual(state.rejected_kv, 4 - accepted)
            self.assertEqual(state.newly_historical, (64, 64 + accepted))
        self.assertEqual(WindowTransition(8, 9, 1, 64, 256).newly_historical, (64, 64))

    def test_request_slots_survive_row_reorder_and_pause(self):
        table = RequestSlots(2)
        a, b = table.begin("a"), table.begin("b")
        self.assertEqual(table.rows(["b", "a"]), (b, a))
        self.assertEqual(table.rows(["b"]), (b,))
        self.assertEqual(table.rows(["a"]), (a,))
        with self.assertRaisesRegex(RuntimeError, "no silent eviction"):
            table.begin("c")

    def test_finished_and_new_same_id_have_new_epoch(self):
        table = RequestSlots(1)
        before = table.begin("same")
        table.retire(["same"])
        after = table.begin("same")
        self.assertEqual(before.index, after.index)
        self.assertGreater(after.epoch, before.epoch)
        resumed = table.begin("same", resumed=True)
        self.assertGreater(resumed.epoch, after.epoch)


if __name__ == "__main__":
    unittest.main()
