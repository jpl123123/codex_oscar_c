import json
from pathlib import Path
import re
import shlex
import tempfile
import unittest
from unittest.mock import patch

from oscar_ascend.service_config import target_argv, task_environment

ROOT = Path(__file__).resolve().parents[1]


class ServiceConfigTests(unittest.TestCase):
    def test_matches_every_appendix_argument(self):
        text = (ROOT / "oscar_ascend_agent_start.md").read_text()
        appendix = text.split("## 附录 A：原始目标启动命令", 1)[1].split("## 附录 B：", 1)[0]
        source = re.search(r"```bash\n(.*?)\n```", appendix, re.S).group(1)
        expected = shlex.split(source.replace("\\\n", " "))
        self.assertEqual(target_argv(ROOT / "configs/target_service.json"), expected)

    def test_rejects_disabled_mtp_and_global_eager(self):
        original = json.loads((ROOT / "configs/target_service.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.json"
            for mutation in (lambda a: a + ["--enforce-eager"],
                             lambda a: ["{\"num_speculative_tokens\":0}" if x.startswith('{"method"') else x for x in a]):
                with self.subTest(mutation=mutation):
                    data = dict(original, argv=mutation(original["argv"]))
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        target_argv(path)

    def test_device_selection_cannot_leak_from_parent(self):
        with patch.dict("os.environ", {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"}):
            self.assertEqual(task_environment(enabled=True)["ASCEND_RT_VISIBLE_DEVICES"], "4,5,6,7")


if __name__ == "__main__":
    unittest.main()
