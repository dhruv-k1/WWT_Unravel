# SASRec + CatBoost Hybrid Recommender

This project trains a hybrid next-item recommender using:
- **SASRec** (Transformer for sequential patterns)
- **Co-occurrence / Popularity / Simple category** features
- **CatBoostRanker (YetiRank)** for reranking to top-K

The notebook includes a **5% group hold-out** (by `CUSTOMER_ID`) to report **Recall@3** and candidate coverage.

## Files
- `SASRec_CatBoost_Ranker_with_Validation.ipynb` — end-to-end, well-documented notebook
- `requirements.txt` — dependencies
- `artifacts/` (created on run) — saved models & features
- `sasrec_catboost_recommendations.csv` — test predictions (written on run)

## Data Assumptions
Place these four CSVs under `DATA_DIR` (default: `/datasets`):
- `order_data.csv`
- `customer_data.csv`
- `store_data.csv`
- `test_data_question.csv`

> Adjust `DATA_DIR` at the top of **Section 2** if running locally.

## Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook and run all cells (GPU recommended for PyTorch and CatBoost).
   - To force CPU: ensure `torch.cuda.is_available()` is False or ignore warnings.

## Outputs
- **Validation metrics** printed at the end:
  - `Queries`: number of validation queries evaluated
  - `Candidate coverage`: fraction where the true item is in the candidate set
  - `Recall@3`: hit ratio at 3 after reranking
- **Test predictions**: `sasrec_catboost_recommendations.csv` with columns:
  - `CUSTOMER_ID, ORDER_ID, item1, item2, item3, Recommendation 1, Recommendation 2, Recommendation 3`

## Customization
- Increase `CFG["SASREC_EPOCHS"]` to 10–15 for stronger sequence modeling.
- Increase `CFG["CATBOOST_ITERS"]` (e.g., 1000) for better ranking, given time.
- Change `CFG["BASE_CAND_N"]`, `SASREC_BLEND_TOP`, `PREPOOL_GLOB` to adjust candidate pool behavior.

## Notes
- All **artifacts are trained only on the train split** to avoid leakage.
- The validation is **grouped by user** so that users don’t overlap between train and validation.
- This is a **lightweight** SASRec; you can add more layers, dropout tuning, or positional variants later.
