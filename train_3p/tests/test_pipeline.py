"""
tests/test_pipeline.py – Unit tests for run_pipeline.py helpers.

Covers:
  * _tail_jsonl  – streaming tail that keeps the last N lines
  * _merge_jsonl – safe concatenation that handles dst ∈ files (rolling buffer)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import _tail_jsonl, _merge_jsonl


# ---------------------------------------------------------------------------
# _tail_jsonl
# ---------------------------------------------------------------------------

class TestTailJsonl:
    def test_tail_keeps_last_n(self, tmp_path):
        src = tmp_path / "src.jsonl"
        dst = tmp_path / "dst.jsonl"
        lines = [json.dumps({"i": i}) + "\n" for i in range(20)]
        src.write_text("".join(lines), encoding="utf-8")

        _tail_jsonl(src, dst, keep_lines=5)

        out = dst.read_text(encoding="utf-8").splitlines()
        assert len(out) == 5
        vals = [json.loads(l)["i"] for l in out]
        assert vals == list(range(15, 20))

    def test_tail_fewer_lines_than_keep(self, tmp_path):
        src = tmp_path / "src.jsonl"
        dst = tmp_path / "dst.jsonl"
        lines = [json.dumps({"i": i}) + "\n" for i in range(3)]
        src.write_text("".join(lines), encoding="utf-8")

        _tail_jsonl(src, dst, keep_lines=10)

        out = dst.read_text(encoding="utf-8").splitlines()
        assert len(out) == 3

    def test_tail_missing_src_is_noop(self, tmp_path):
        dst = tmp_path / "dst.jsonl"
        _tail_jsonl(tmp_path / "nonexistent.jsonl", dst, keep_lines=5)
        assert not dst.exists()

    def test_tail_large_file_memory_efficient(self, tmp_path):
        """deque-based tail should work on a 10 000-line file."""
        src = tmp_path / "big.jsonl"
        with src.open("w", encoding="utf-8") as f:
            for i in range(10_000):
                f.write(json.dumps({"i": i}) + "\n")
        dst = tmp_path / "tail.jsonl"

        _tail_jsonl(src, dst, keep_lines=100)

        out = dst.read_text(encoding="utf-8").splitlines()
        assert len(out) == 100
        assert json.loads(out[-1])["i"] == 9999


# ---------------------------------------------------------------------------
# _merge_jsonl
# ---------------------------------------------------------------------------

class TestMergeJsonl:
    def _make_jsonl(self, path: Path, values: list) -> Path:
        with path.open("w", encoding="utf-8") as f:
            for v in values:
                f.write(json.dumps({"v": v}) + "\n")
        return path

    def test_merge_two_files(self, tmp_path):
        a = self._make_jsonl(tmp_path / "a.jsonl", [1, 2, 3])
        b = self._make_jsonl(tmp_path / "b.jsonl", [4, 5, 6])
        dst = tmp_path / "out.jsonl"

        _merge_jsonl([a, b], dst)

        vals = [json.loads(l)["v"] for l in dst.read_text().splitlines()]
        assert vals == [1, 2, 3, 4, 5, 6]

    def test_merge_with_keep_last(self, tmp_path):
        a = self._make_jsonl(tmp_path / "a.jsonl", list(range(10)))
        b = self._make_jsonl(tmp_path / "b.jsonl", list(range(10, 20)))
        dst = tmp_path / "out.jsonl"

        _merge_jsonl([a, b], dst, keep_last=5)

        vals = [json.loads(l)["v"] for l in dst.read_text().splitlines()]
        assert len(vals) == 5
        assert vals == list(range(15, 20))

    def test_merge_rolling_buffer_preserves_prior_data(self, tmp_path):
        """
        Critical: _merge_jsonl([replay, new_data], replay, keep_last=…)
        must not lose prior replay data by truncating dst before reading it.
        """
        replay = self._make_jsonl(tmp_path / "replay.jsonl", list(range(10)))
        new_data = self._make_jsonl(tmp_path / "new.jsonl", list(range(10, 15)))

        _merge_jsonl([replay, new_data], replay, keep_last=20)

        vals = [json.loads(l)["v"] for l in replay.read_text().splitlines()]
        # All 15 values should be present (below keep_last threshold)
        assert len(vals) == 15
        assert vals == list(range(15))

    def test_merge_rolling_buffer_truncation(self, tmp_path):
        """When combined size > keep_last, only the tail is kept."""
        replay = self._make_jsonl(tmp_path / "replay.jsonl", list(range(8)))
        new_data = self._make_jsonl(tmp_path / "new.jsonl", list(range(8, 12)))

        _merge_jsonl([replay, new_data], replay, keep_last=5)

        vals = [json.loads(l)["v"] for l in replay.read_text().splitlines()]
        assert len(vals) == 5
        assert vals == list(range(7, 12))

    def test_merge_missing_file_skipped(self, tmp_path):
        a = self._make_jsonl(tmp_path / "a.jsonl", [1, 2])
        dst = tmp_path / "out.jsonl"

        _merge_jsonl([a, tmp_path / "missing.jsonl"], dst)

        vals = [json.loads(l)["v"] for l in dst.read_text().splitlines()]
        assert vals == [1, 2]

    def test_merge_no_keep_last(self, tmp_path):
        a = self._make_jsonl(tmp_path / "a.jsonl", [10, 20, 30])
        dst = tmp_path / "out.jsonl"

        _merge_jsonl([a], dst)

        vals = [json.loads(l)["v"] for l in dst.read_text().splitlines()]
        assert vals == [10, 20, 30]
