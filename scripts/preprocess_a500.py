"""
Preprocesses the A500 (Chinese A-share) per-ticker data into the same format as data/SP100/raw/,
so the existing PyG dataset (SP100Stocks) and notebooks work on it unchanged.

Input:  data/A500_RAW/<code>.csv — one file per ticker with columns date, code, market,
        sector_name, open, high, low, close, volume, ... (directory link to the
        QuantTraderAlpha101 A500 data).
Output: data/A500/raw/values.csv  — (Symbol, Date) indexed features, same columns as SP100:
                                    Close, NormClose, DailyLogReturn, ALR1W, ALR2W, ALR1M, ALR2M, RSI, MACD
        data/A500/raw/stocks.csv  — Symbol, Name, Sector
        data/A500/raw/adj.npy     — adjacency matrix

Feature engineering mirrors notebooks/1-data_collection_and_preprocessing.ipynb exactly.
The A500 raw data has no fundamentals, so the adjacency (notebook 2) is built from the
correlation of daily log returns instead of fundamentals correlation, keeping the same
sector-bonus merge. The correlation threshold is chosen so the mean node degree matches
the SP100 graph (~5-6) unless --corr-threshold is given explicitly.

Usage (from the repo root):
	uv run python scripts/preprocess_a500.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from ta.momentum import rsi
from ta.trend import macd

FEATURE_COLUMNS = ["Close", "NormClose", "DailyLogReturn", "ALR1W", "ALR2W", "ALR1M", "ALR2M", "RSI", "MACD"]


def load_ticker(path: str) -> tuple[str, str, pd.DataFrame]:
	"""
	Loads one raw ticker CSV.
	:param path: Path of the raw per-ticker CSV
	:return: (symbol, sector, dataframe of close prices indexed by date)
	"""
	df = pd.read_csv(path, usecols=["date", "code", "market", "sector_name", "close"], dtype={"code": str})
	symbol = f"{df['code'].iloc[-1]}.{df['market'].iloc[-1]}"  # e.g. 000001.SZ — keeps Symbol a string in CSVs
	sector = df["sector_name"].iloc[-1]
	df = df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
	return symbol, sector, df[["close"]].rename(columns={"close": "Close"})


def compute_features(close: pd.Series) -> pd.DataFrame:
	"""
	Computes the SP100 feature set on a full close-price history (same as notebook 1).
	Normalizations that depend on the window (NormClose, return stds) are applied later,
	after slicing to the common calendar.
	:param close: Close prices indexed by date
	:return: DataFrame with un-normalized features
	"""
	df = pd.DataFrame({"Close": close})
	df["DailyLogReturn"] = np.log(1 + df["Close"].pct_change())
	df["ALR1W"] = df["DailyLogReturn"].rolling(window=5).sum() * 5
	df["ALR2W"] = df["DailyLogReturn"].rolling(window=10).sum() * 5
	df["ALR1M"] = df["DailyLogReturn"].rolling(window=21).sum() * 21
	df["ALR2M"] = df["DailyLogReturn"].rolling(window=42).sum() * 21
	df["RSI"] = rsi(df["Close"]) / 100
	df["MACD"] = macd(df["Close"])
	return df


def normalize_window(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Applies the window-dependent normalizations of notebook 1 on the sliced data.
	:param df: Features of one ticker, restricted to the target window
	:return: Normalized features, in the SP100 column order
	"""
	df = df.copy()
	df["NormClose"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()
	for col in ["DailyLogReturn", "ALR1W", "ALR2W", "ALR1M", "ALR2M"]:
		df[col] /= df[col].std()
	return df[FEATURE_COLUMNS]


def build_adjacency(returns: pd.DataFrame, sectors: pd.Series, corr_threshold: float, sector_bonus: float, target_degree: float) -> np.ndarray:
	"""
	Builds the adjacency matrix as in notebook 2, with daily-return correlation replacing
	the fundamentals correlation (the A500 raw data has no fundamentals).
	:param returns: Daily log returns (dates x symbols)
	:param sectors: Sector of each symbol
	:param corr_threshold: Fixed correlation threshold; if None it is chosen to reach target_degree
	:param sector_bonus: Correlation bonus for two stocks sharing the same sector
	:param target_degree: Desired mean node degree when corr_threshold is None
	:return: The (n, n) adjacency matrix
	"""
	corr = returns.corr().to_numpy().copy()
	np.fill_diagonal(corr, 0)
	share_sector = (sectors.to_numpy()[:, None] == sectors.to_numpy()[None, :]).astype(int) - np.eye(len(sectors), dtype=int)
	scores = abs(corr) + share_sector * sector_bonus  # abs because GCNConv only accepts positive weights
	if corr_threshold is None:
		off_diag = scores[~np.eye(len(scores), dtype=bool)]
		corr_threshold = float(np.quantile(off_diag, 1 - target_degree / (len(scores) - 1)))
	adj = scores * (scores > corr_threshold)
	adj = adj / adj.max()
	print(f"Adjacency: threshold={corr_threshold:.4f}, edges={np.count_nonzero(adj)}, mean degree={np.count_nonzero(adj) / len(adj):.2f}")
	return adj


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--raw-dir", default="data/A500_RAW", help="Directory of per-ticker CSVs")
	parser.add_argument("--out-dir", default="data/A500/raw", help="Output directory (SP100-like raw folder)")
	parser.add_argument("--years", type=float, default=5.0, help="Length of the historical window in years")
	parser.add_argument("--min-coverage", type=float, default=0.95, help="A trading date is kept in the calendar if at least this share of tickers has it")
	parser.add_argument("--corr-threshold", type=float, default=None, help="Fixed correlation threshold for edges (default: auto from --target-degree)")
	parser.add_argument("--sector-bonus", type=float, default=0.05, help="Correlation bonus for same-sector stocks (as notebook 2)")
	parser.add_argument("--target-degree", type=float, default=5.5, help="Desired mean node degree when --corr-threshold is not given")
	args = parser.parse_args()

	files = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))
	assert files, f"No CSV files found in {args.raw_dir}"
	print(f"Loading {len(files)} tickers from {args.raw_dir}...")
	tickers: dict[str, pd.DataFrame] = {}
	sectors: dict[str, str] = {}
	for path in files:
		symbol, sector, df = load_ticker(path)
		tickers[symbol] = df
		sectors[symbol] = sector

	# Common trading calendar over the last `years`: dates present in at least min_coverage of tickers
	last_date = max(df.index[-1] for df in tickers.values())
	start_date = (pd.Timestamp(last_date) - pd.DateOffset(days=round(args.years * 365.25))).strftime("%Y-%m-%d")
	date_counts = pd.concat([df.loc[df.index >= start_date].index.to_series() for df in tickers.values()]).value_counts()
	calendar = date_counts[date_counts >= args.min_coverage * len(tickers)].index.sort_values()
	print(f"Calendar: {len(calendar)} trading days from {calendar[0]} to {calendar[-1]}")

	# Keep tickers that cover the full calendar, compute features on full history, slice, normalize
	values, kept = [], []
	for symbol, df in tickers.items():
		if not calendar.isin(df.index).all():
			continue
		features = compute_features(df["Close"]).loc[calendar]
		features = normalize_window(features)
		if features.isna().any().any():  # e.g. listed too recently for the rolling warm-up
			continue
		values.append(features)
		kept.append(symbol)
	print(f"Kept {len(kept)}/{len(tickers)} tickers with complete data on the calendar")

	values = pd.concat(values, keys=kept, names=["Symbol", "Date"])

	adj = build_adjacency(
		returns=values["DailyLogReturn"].unstack(level="Symbol")[kept],
		sectors=pd.Series([sectors[s] for s in kept], index=kept),
		corr_threshold=args.corr_threshold,
		sector_bonus=args.sector_bonus,
		target_degree=args.target_degree,
	)

	stocks = pd.DataFrame({"Symbol": kept, "Name": kept, "Sector": [sectors[s] for s in kept]}).set_index("Symbol")

	os.makedirs(args.out_dir, exist_ok=True)
	values.to_csv(os.path.join(args.out_dir, "values.csv"), encoding="utf-8")
	stocks.to_csv(os.path.join(args.out_dir, "stocks.csv"), encoding="utf-8")
	np.save(os.path.join(args.out_dir, "adj.npy"), adj)
	print(f"Wrote values.csv {values.shape}, stocks.csv {stocks.shape}, adj.npy {adj.shape} to {args.out_dir}")


if __name__ == "__main__":
	main()
