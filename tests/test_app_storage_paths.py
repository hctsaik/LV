import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from app_storage_paths import read_jsonl_logs, resolve_output_dir


class AppStoragePathsTests(unittest.TestCase):
    def test_output_dir_uses_managed_path_and_keeps_standalone_default(self):
        source = Path("source") / "output"
        managed = Path("store") / "data" / "visuallatent" / "output"
        self.assertEqual(resolve_output_dir(source, {"LV_OUTPUT_DIR": str(managed)}), managed)
        self.assertEqual(resolve_output_dir(source, {}), source)

    def test_curation_reader_merges_store_and_legacy_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory(prefix="lv-app-storage-") as temporary:
            root = Path(temporary)
            legacy = root / "source" / "curation_log.jsonl"
            managed = root / "store" / "curation_log.jsonl"
            legacy.parent.mkdir(parents=True)
            managed.parent.mkdir(parents=True)
            same_ts = "2026-10-18 10:00:00"
            legacy.write_text(
                json.dumps({"ts": same_ts, "id": "legacy-first"}) + "\n"
                + json.dumps({"ts": same_ts, "id": "legacy-later"}) + "\n"
                + "not-json\n", encoding="utf-8")
            managed.write_text(
                json.dumps({"ts": same_ts, "id": "managed"}) + "\n",
                encoding="utf-8")

            rows = read_jsonl_logs((legacy, managed, legacy), newest_first=True)

            self.assertEqual(
                [row["id"] for row in rows],
                ["managed", "legacy-later", "legacy-first"],
            )


if __name__ == "__main__":
    unittest.main()
