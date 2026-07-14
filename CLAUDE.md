# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of S&P100 stocks using Graph Neural Networks: forecasting, clustering, trend classification, and portfolio optimization (including a DRL/PPO approach). The project is driven by nine numbered Jupyter notebooks in `notebooks/`, backed by shared Python modules in the same directory.

## Commands

The venv is managed with uv (`.venv/` at the repo root, Python 3.11):

```powershell
uv venv --python 3.11                 # create venv (once)
uv pip install -r requirements.txt    # install deps (PyTorch, PyTorch Geometric, gymnasium, yfinance, ...)
uv run jupyter notebook               # run notebooks; there are no tests, linters, or build steps
uv run tensorboard --logdir notebooks/runs   # view training curves (train() logs here)
```

torch must stay <2.6 (pinned in requirements.txt): newer torch defaults `torch.load` to `weights_only=True`, which breaks `SP100Stocks.get()` loading pickled PyG `Data` objects.

## How the pieces fit together

**Notebooks are a pipeline.** Notebooks 1 and 2 produce the raw data in `data/SP100/raw/` (`values.csv` — per-stock daily features indexed by (Symbol, Date); `adj.npy` — the adjacency matrix built from sector + fundamentals correlation; `stocks.csv`, `fundamentals.csv`). These raw files are checked into git, so later notebooks work without re-running 1–2. Notebooks 3–9 build on them: 3 explains the PyG dataset, 4 the temporal GNN models, 5 forecasting, 6 clustering, 7 DRL weight optimization, 8 trend classification, 9 portfolio selection (uses the classifier from 8).

**Working directory and imports.** Notebooks run with `notebooks/` as the CWD but import shared code via the package path `notebooks.models` / `notebooks.datasets` / `notebooks.ppo` (repo root must be on `sys.path`). Data paths in both notebooks and modules are relative to `notebooks/` (e.g. `../data/SP100/raw/values.csv`) — `SP100Stocks` even hardcodes them in `process()`. Keep both conventions intact when editing.

**Dataset layer** (`notebooks/datasets/`): `SP100Stocks` is a PyG `Dataset` that slices the full time series into one `Data` object per timestep, each with a `past_window` of features and a `future_window` target, saved as `timestep_{t}.pt` under `data/SP100/processed/`. Processed dirs (`processed/`, `forecasting_processed/`) are gitignored (~8GB each) and regenerated on first use; pass `force_reload=True` after changing windows or transforms. Tensor conventions from `get_graph_in_pyg_format`: `x` is `(nodes, features, time)` with feature 0 being the (normalized) close price; raw `close_price` is carried separately on each `Data` for computing returns. Tasks are re-targeted via PyG `transform` functions that rewrite `sample.y` (e.g. notebook 8 turns future close prices into binary buy/sell classes relative to market return).

**Model layer** (`notebooks/models/`): temporal GNNs — `TGCN` (stack of `TGCNCell`, a GRU whose gates use graph convolutions; `use_gat=True` swaps GCN for GAT), `A3TGCN` (attention over T-GCN hidden states), and `DCGNN` (built on `DCGRUCell`, diffusion convolution). All share the forward signature `model(x, edge_index, edge_weight)` operating on a single graph (no batch dimension over nodes) and return `(nodes, out_features)`. `train.py:train()` is the shared loop for both regression and classification (pass `measure_acc=True` for classification) and logs to TensorBoard under `notebooks/runs/`. Evaluation helpers live in `evaluate.py`.

**RL layer** (`notebooks/ppo/`): a PPO implementation adapted for PyG graph observations (gymnasium env whose observations are batched `Data` objects), used only by notebook 7. Note the README caveat: notebook 7 was never fully trained for lack of compute; the code is reference-quality but functional.

## Conventions

- Python files use tabs for indentation and docstrings with `:param:` style.
- Model/module code lives in `.py` files; notebooks demonstrate and orchestrate. Mirror changes in both when a notebook re-implements module logic for exposition (e.g. notebook 3 vs `SP100Stocks.process`).
