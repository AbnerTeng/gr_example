import json
import tempfile
from pathlib import Path

from src.prep_nq import write_oracle_ceiling


def test_oracle_summary_is_written_under_out_dir():
    with tempfile.TemporaryDirectory() as directory:
        out_dir = Path(directory) / "nq320k"
        result = {"gold_recall@1": 0.5}
        path = write_oracle_ceiling(out_dir, result)
        assert path == out_dir / "oracle_ceiling.json"
        assert json.loads(path.read_text()) == result


if __name__ == "__main__":
    test_oracle_summary_is_written_under_out_dir()
    print("PASS test_oracle_summary_is_written_under_out_dir")
