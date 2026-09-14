import unittest
from oscar_ascend.compatibility import check_version_families, require_parameters, version_family


class CompatibilityTests(unittest.TestCase):
    def test_accepts_declared_development_and_local_versions(self):
        result = check_version_families("0.23.0+empty", "0.23.1.dev0+g5cb98caaa.d20260822")
        self.assertEqual(result["ascend_release"], (0, 23, 1))
        self.assertEqual(result["status"], "version_family_matches_only")

    def test_rejects_different_release_or_invalid_version(self):
        with self.assertRaises(RuntimeError):
            check_version_families("0.24.0", "0.23.0")
        with self.assertRaises(ValueError):
            version_family("unknown-build")

    def test_interface_presence_is_checked_independently(self):
        def changed(self, renamed):
            return None
        with self.assertRaisesRegex(RuntimeError, "Native interface changed"):
            require_parameters(changed, {"self", "scheduler_output"}, "runner")


if __name__ == "__main__":
    unittest.main()
