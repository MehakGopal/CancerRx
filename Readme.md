# Cancer Drug Sensitivity Prediction from Gene Expression

**Status: in progress.** Numbers below are validation-set only — test set is held out and unused so far (see Roadmap).

## Problem statement

Predict a cancer cell line's drug sensitivity (LN_IC50, log-scale IC50 from GDSC2) from its gene expression profile. Framed per-drug: one model per drug, trained on that drug's cell lines only, since mechanisms differ enough that a shared model would blur interpretability without a clear accuracy win at this sample size (~650–950 cell lines per drug).

## Drugs chosen

| Drug | Target | Notes |
|------|--------|-------|
| Erlotinib | EGFR (TKI) | Primary benchmark |
| Gefitinib | EGFR (TKI) | Replication check — same target as Erlotinib, different compound |
| Paclitaxel | Tubulin / mitosis | Broader mechanism, weaker single-target signal expected |
| Tamoxifen | ESR1 (estrogen receptor) | Strong signal expected in ER+ lines |
| Gemcitabine | DNA replication (nucleoside analog) | Metabolic-gene driven |

## Data and preprocessing 
Built in `notebooks/explore_data.ipynb` and `notebooks/gsva.r`.

- **Drug response:** — LN_IC50/AUC per cell line per drug, filtered to the 5 drugs above.
- **Expression:** RSEM-processed RNA-seq log normalized TPM scores merged across cell lines, transposed to cell-line rows.
- **Join:** expression joined to drug response on `SANGER_MODEL_ID` 
- **Pathway enrichment:**  runs GSVA (Gaussian kernel) on log-TPM expression against MSigDB gene sets, producing per-cell-line pathway scores. Two collections tried so far: `C2:CP:PID` (~196 pathways) and `Hallmark` (~50 pathways).
- **Merge + split:** pathway scores merged back with drug-response metadata (`src/data.py::build_pathway_enriched_data`), then split 70/10/20 train/val/test, **grouped by `model_id` (cell line) independently per drug**, so no cell line leaks across splits (`src/data.py::make_splits`, leakage-checked with an assert).

## Feature selection

Raw expression is ~41k genes — too high-dimensional and noisy for ~650 samples per drug. Instead of selecting genes directly, we use **GSVA pathway enrichment scores** as features (one score per MSigDB pathway per cell line), per the rationale in [Fustero-Torre et al., *Nat Commun* 2022](https://www.nature.com/articles/s41467-022-33291-z.pdf): pathway-level scores are more generalizable and less noisy than single genes at small sample sizes.

Two pathway collections compared head-to-head so far: `PID_` (C2:CP:PID) and `HALLMARK_`. `PID_` wins on 4 of 5 drugs (see Best results).

However, the R2 scores aren't the best, especially Erlotinib which has shown better scores in different papers. Training utilising genes is in the roadmap.

## Models

Two stages, both trained per-drug independently, all runs logged automatically (see below):

- **Baseline** (`src/models.py::baseline_models`): ElasticNetCV (5-fold), dummy mean regressor.
- **Candidates** (`src/models.py::candidate_models`): decision tree, random forest, SVM, MLP, XGBoost, LightGBM — all hand-constrained against overfitting given ~650 train rows (e.g. `max_depth`, `min_samples_leaf`, MLP `alpha`/early stopping, boosted-tree learning rate/depth). XGBoost and LightGBM use validation-set early stopping.

Every training run (`src/train.py::run_stage`) fits the full model set across all 5 drugs and appends results to `results/metrics.csv` (full history, tagged with `run_id`, `stage`, `feature_prefix`, timestamp) and `results/best_models.csv` (best-r² model per drug, also append-only). Nothing lives only in notebook memory anymore.

## Best results

Best model per drug, best of `PID_` vs `HALLMARK_` feature sets, **validation set**:

| Drug | Feature set | Model | RMSE | r² |
|------|------------|-------|------|-----|
| Gemcitabine | PID_ | mlp | 2.260 | 0.401 |
| Tamoxifen | PID_ | elasticnet | 1.108 | 0.307 |
| Paclitaxel | HALLMARK_ | mlp | 1.494 | 0.306 |
| Gefitinib | PID_ | svm | 0.938 | 0.261 |
| Erlotinib | PID_ | lightgbm | 1.257 | 0.181 |

## Roadmap

1. **Try gene-level feature selection instead of (or alongside) pathway scores** — on raw expression, compare against the GSVA pathway-score approach. 
2. **Final test-set evaluation** — everything above is validation-set only; lock in real numbers on the held-out test split for the winning model per drug.
3. **SHAP / feature interpretation** — tie top pathways (or genes, depending on (1)) back to known drug mechanism per drug.

## Literature survey

- Fustero-Torre et al. (2022). Pathway-based drug response prediction — motivates using pathway enrichment scores over raw genes. [*Nat Commun*](https://www.nature.com/articles/s41467-022-33291-z.pdf)
- [RPKM/FPKM/TPM explained](https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/) — background on RNA-seq normalization units used in preprocessing.

*More references to be added.*
