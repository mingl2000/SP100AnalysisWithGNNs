"""
Clones the SP100 notebooks into *_a500.ipynb variants that run on data/A500/ instead
of data/SP100/. Re-run this script to regenerate the clones after editing the originals.

Notebooks 1-2 are not cloned (scripts/preprocess_a500.py replaces them for A500) and
notebook 4 is not cloned (pure model implementation, no data references).

Usage (from the repo root):
	uv run python scripts/clone_notebooks_a500.py
"""
import glob
import json
import re

NOTEBOOKS = [3, 5, 6, 7, 8, 9]

REPLACEMENTS = [  # (regex, replacement), applied to every cell in order
	(r"\.\./data/SP100/", "../data/A500/"),
	(r"SP100Stocks\(\)", 'SP100Stocks(root="../data/A500/")'),
	(r"SP100Stocks\((?!root)(?=\w)", 'SP100Stocks(root="../data/A500/", '),
	(r"get_stocks_labels\(\)", 'get_stocks_labels("../data/A500/raw/values.csv")'),
	(r"UpDownTrend", "A500_UpDownTrend"),  # saved model files and run titles of notebooks 8/9
	(r"runs/StocksClustering_", "runs/A500_StocksClustering_"),
	(r"S&P ?100", "A500"),
]

BANNER = (
	"**A500 variant** — generated from the SP100 notebook by `scripts/clone_notebooks_a500.py`; "
	"do not edit directly, edit the original and re-run the script.\n\n"
	"Requires `data/A500/raw/` built by `scripts/preprocess_a500.py` (replaces notebooks 1-2 for A500)."
)


def clone(path: str) -> str:
	nb = json.load(open(path, encoding="utf-8"))
	for cell in nb["cells"]:
		src = "".join(cell["source"])
		for pattern, repl in REPLACEMENTS:
			src = re.sub(pattern, repl, src)
		cell["source"] = src.splitlines(keepends=True)
		if cell["cell_type"] == "code":  # clear stale SP100 outputs
			cell["outputs"], cell["execution_count"] = [], None
	nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": BANNER.splitlines(keepends=True)})
	out_path = path.replace(".ipynb", "_a500.ipynb")
	json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1)  # ensure_ascii keeps the file readable regardless of locale encoding
	return out_path


if __name__ == "__main__":
	for n in NOTEBOOKS:
		src_path = next(p for p in glob.glob(f"notebooks/{n}-*.ipynb") if not p.endswith("_a500.ipynb"))
		print(f"{src_path} -> {clone(src_path)}")
