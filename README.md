# Telco Customer Churn API (Notebook-first Phase 1)

Phase 1 is implemented **inside the provided notebook**:

- `notebooks/telco_churn_ordered_sota_modeling.ipynb`

A new section was added to the notebook:

- `## 21) Phase 1 Export for API Deployment`

When you run the notebook through the model-selection cells and then execute section 21,
it exports deployment artifacts to `artifacts/`:

- `artifacts/churn_model.joblib`
- `artifacts/metadata.json`
- `artifacts/leaderboard.json`
- `artifacts/sample_payload.json`

## How to use

1. Open and run `notebooks/telco_churn_ordered_sota_modeling.ipynb` from top to bottom.
2. Execute section `21) Phase 1 Export for API Deployment`.
3. Verify the files above exist under `artifacts/`.

## Notes

- Export requires notebook objects from prior sections (`best_obj`, `metrics_df`, `optimal_row`, etc.).
- Export cell includes path auto-detection for project root by searching for `data/telco.csv`.
