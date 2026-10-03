"""One-time mechanical extraction from the user's immutable research archive.

Not required for inference or figure regeneration after this snapshot is committed.
Records hashes and function names rather than exporting unrelated Lorentzian code.
"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(source):
    source = Path(source)
    dest = ROOT / "src/ebm_audit"
    origin = source / "src/intraday_ml"
    provenance = {}

    def extract(filename, target, names=None, header=""):
        path = origin / filename
        text = path.read_text()
        if names:
            lines = text.splitlines(keepends=True)
            selected = [
                n for n in ast.parse(text).body if getattr(n, "name", None) in names
            ]
            assert len(selected) == len(names)
            text = header + "\n\n".join(
                "".join(lines[n.lineno - 1 : n.end_lineno]) for n in selected
            )
        (dest / target).write_text(text)
        provenance[target] = {
            "source": f"src/intraday_ml/{filename}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "functions": names or "whole file",
        }

    for old, new in [
        ("v3_runner.py", "runner.py"),
        ("v5_execution.py", "execution.py"),
    ]:
        extract(old, new)
    extract(
        "data.py", "data.py",
        ["checksum", "make_features", "make_events", "write_json"],
        (origin / "data.py").read_text().split("\ndef checksum(")[0] + "\n\n",
    )
    extract(
        "trading.py", "trading.py",
        ["slippage", "commission_unit", "directions", "metrics"],
        "import numpy as np\n\n",
    )
    extract(
        "v3_run.py",
        "validation.py",
        ["audit_trade"],
        "import numpy as np\nimport pandas as pd\n\n",
    )
    extract(
        "v5_run.py",
        "controls.py",
        ["compare_execution", "eligible_paths", "random_plans", "controls"],
        "import numpy as np\nimport pandas as pd\nfrom .data import write_json\n"
        "from .runner import account_scenario, complete_session_path, simulate\n"
        "from .validation import audit_trade\n"
        "from .execution import QUOTE_COLUMNS, compressed_accounts, path_summary\n"
        "MONTHS = pd.period_range('2014-01', '2020-12', freq='M').astype(str).tolist()\n\n",
    )
    cpp = (origin / "v5_ann.cpp").read_text()
    marker = "// Exact execution cache accelerator;"
    (dest / "execution.cpp").write_text(
        "#include <cmath>\n#include <limits>\n#include <algorithm>\n"
        + marker
        + cpp.split(marker)[1]
    )
    provenance["execution.cpp"] = {
        "source": "src/intraday_ml/v5_ann.cpp",
        "sha256": hashlib.sha256((origin / "v5_ann.cpp").read_bytes()).hexdigest(),
        "functions": ["trade_path (ANN algorithm excluded)"],
    }
    (ROOT / "config").mkdir(exist_ok=True)
    cfg = json.loads((source / "config/experiment_v3_runner.json").read_text())
    cfg["models"] = ["ebm"]
    cfg.update(
        random_replicates=2000,
        random_seed=261004,
        bootstrap_draws=2000,
        bootstrap_seed=261005,
        bootstrap_block_sessions=5,
    )
    cfg["source_spec"] = json.loads(
        (source / "config/experiment_v2_spy.json").read_text()
    )["source_spec"]
    (ROOT / "config/audit.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for fold in range(1, 8):
        directory = ROOT / f"models/fold{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            source / f"artifacts/spy_v2/models/fold{fold}/ebm/portable.json",
            directory / "portable.json",
        )
    (ROOT / "docs/SOURCE_SNAPSHOT.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    main(parser.parse_args().source)
