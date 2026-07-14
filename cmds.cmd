cd D:\PriProjects\RTGnn\SP100AnalysisWithGNNs\notebooks
$env:PYTHONPATH = "D:\PriProjects\RTGnn\SP100AnalysisWithGNNs"
uv run jupyter nbconvert --to notebook --execute --inplace 8-stock_trend_classification.ipynb