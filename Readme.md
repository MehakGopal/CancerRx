# cancer-rx: Cancer Drug Sensitivity Prediction from Gene Expression

**Status: in progress.** All numbers below are validation-set only. The test set is held out and has not been used yet (see Roadmap).

## Problem statement

Predict a cancer cell line's drug sensitivity (LN_IC50, log-scale IC50 from GDSC2) from its gene expression profile.

The problem is framed per drug: one model per drug, trained only on that drug's cell lines. Drug mechanisms differ enough that a shared model would blur interpretability, without a clear accuracy gain at this sample size (~930 cell lines per drug, ~650 for training).

Expression is measured *before* treatment. The models learn which baseline pathway states are associated with sensitivity, not drug-induced changes.

## Drugs chosen

| Drug | Target | Notes |
|------|--------|-------|
| Erlotinib | EGFR (TKI) | Primary benchmark |
| Gefitinib | EGFR (TKI) | Replication check: same target as Erlotinib, different compound |
| Paclitaxel | Tubulin / mitosis | Broader mechanism, weaker single-target signal expected |
| Tamoxifen | ESR1 (estrogen receptor) | Strong signal expected in ER+ lines |
| Gemcitabine | DNA replication (nucleoside analog) | Metabolic-gene driven |

## Data and preprocessing

Raw data is **not included** in this repo. Download it from the original sources below and place the files in `data/`.

| Data | Source | File |
|---|---|---|
| Drug response (LN_IC50) | [GDSC bulk downloads](https://www.cancerrxgene.org/downloads/bulk_download) | `GDSC2_fitted_dose_response_27Oct23.xlsx` |
| Expression (RSEM TPM) | [Cell Model Passports downloads](https://cellmodelpassports.sanger.ac.uk/downloads) | `rnaseq_merged_20260323.zip` |

Running `notebooks/explore_data.ipynb` and then `notebooks/gsva.r` generates `required_data.parquet` and the `gsva_*_scores.csv` files.

Built in `notebooks/explore_data.ipynb` and `notebooks/gsva.r`.

- **Drug response:** LN_IC50/AUC per cell line per drug, filtered to the 5 drugs above.
- **Expression:** RSEM-processed RNA-seq, log-normalized TPM, merged across cell lines and transposed to one row per cell line. The values are already log-normalized, so no extra `log2(TPM+1)` is applied.
- **Join:** expression joined to drug response on `SANGER_MODEL_ID`.
- **Pathway enrichment:** GSVA (Gaussian kernel) on log-TPM expression against MSigDB gene sets, giving one score per pathway per cell line.
- **Merge + split:** pathway scores are merged back with drug-response metadata (`src/data.py::build_pathway_enriched_data`), then split 70/10/20 train/val/test. The split is **grouped by `model_id` (cell line), separately for each drug**, so no cell line appears in more than one split (`src/data.py::make_splits`, checked with an assert).
  - Totals: 3,263 / 465 / 935 rows; per drug ~650 / 93 / ~187.

## Feature selection

Raw expression has ~41k genes, which is too high-dimensional and noisy for ~650 training samples per drug. Instead of selecting genes directly, the models use **GSVA pathway enrichment scores** as features. This follows [Fustero-Torre et al., *Nat Commun* 2022](https://www.nature.com/articles/s41467-022-33291-z.pdf): pathway-level scores generalize better and are less noisy than single genes at small sample sizes.

Two MSigDB collections were compared head-to-head:

| Feature set | # pathways | Rationale |
|---|---|---|
| `HALLMARK_` | 50 | Broad, low-redundancy biological processes |
| `PID_` (C2:CP:PID) | 196 | Receptor-level signalling pathways (e.g. `PID_EGFR1_PATHWAY`), which Hallmark lacks |

Reactome (1,692 pathways) was considered and rejected as too many features for the sample size.

## Models

Two stages, each trained independently per drug:

- **Baseline** (`src/models.py::baseline_models`)
  - Dummy mean regressor: the floor every model has to beat.
  - ElasticNetCV (5-fold): a linear reference, used to test whether nonlinear models add anything beyond linear signal.
- **Candidates** (`src/models.py::candidate_models`): decision tree, random forest, SVM, MLP, XGBoost, LightGBM.

With only ~650 training rows, default hyperparameters fail, so every candidate is explicitly constrained:

| Model | Problem with defaults | Fix |
|---|---|---|
| XGBoost | Early-stopped at 1 tree (`learning_rate=0.3`) | `learning_rate=0.03`, subsampling |
| Decision tree | Memorized the training set | `max_depth`, `min_samples_leaf` |
| MLP | Diverged (negative R²) | `alpha=1.0`, `early_stopping=True` |

XGBoost and LightGBM use validation-set early stopping, so their validation R² is slightly optimistic compared with the other models.

Every training run (`src/train.py::run_stage`) fits the full model set across all 5 drugs and appends results to two files:
- `results/metrics.csv`: full history, tagged with `run_id`, `stage`, `feature_prefix` and timestamp.
- `results/best_models.csv`: the best-R² model per drug, also append-only.

Nothing lives only in notebook memory.

## Results

### Best model per drug (validation set)

| Drug | Feature set | Model | RMSE | R² |
|------|------------|-------|------|-----|
| Gemcitabine | PID_ | MLP | 2.260 | 0.401 |
| Tamoxifen | PID_ | ElasticNet | 1.108 | 0.307 |
| Paclitaxel | HALLMARK_ | MLP | 1.494 | 0.306 |
| Gefitinib | PID_ | SVM | 0.938 | 0.261 |
| Erlotinib | PID_ | LightGBM | 1.257 | 0.181 |

RMSE is on the LN_IC50 scale, and that scale's spread differs by drug. Compare drugs using R² only.

### Hallmark vs PID (best model per drug, validation R²)

| Drug | HALLMARK_ | PID_ | Δ R² |
|---|---|---|---|
| Gemcitabine | 0.344 (ElasticNet) | **0.401** (MLP) | +0.057 |
| Erlotinib | 0.113 (XGBoost) | **0.181** (LightGBM) | +0.068 |
| Gefitinib | 0.230 (SVM) | **0.261** (SVM) | +0.031 |
| Tamoxifen | 0.306 (ElasticNet) | 0.307 (ElasticNet) | +0.001 |
| Paclitaxel | **0.306** (MLP) | 0.257 (Random forest) | −0.049 |
| **Mean** | 0.260 | 0.281 | +0.021 |

### All models, HALLMARK_ features (validation R²)

| Model | Gefitinib | Paclitaxel | Erlotinib | Gemcitabine | Tamoxifen |
|---|---|---|---|---|---|
| Dummy (mean) | −0.001 | −0.003 | −0.008 | −0.004 | −0.000 |
| ElasticNet | 0.185 | 0.200 | 0.091 | **0.344** | **0.306** |
| SVM | **0.230** | 0.149 | 0.083 | 0.264 | 0.267 |
| Random forest | 0.202 | 0.220 | 0.088 | 0.326 | 0.277 |
| XGBoost | 0.163 | 0.235 | **0.113** | 0.311 | 0.284 |
| LightGBM | 0.147 | 0.250 | 0.106 | 0.311 | 0.286 |
| MLP | −0.138 | **0.306** | 0.080 | 0.317 | 0.142 |
| Decision tree | −0.063 | 0.108 | −0.217 | 0.115 | 0.078 |

## Key findings

1. **There is real signal.** The mean predictor scores R² ≈ 0 on every drug, and every non-tree model except the MLP on Gefitinib clears it.
2. **Features, not models, set the ceiling.** Linear, tree, boosted and neural models land within a narrow R² band for each drug, and ElasticNet matches or beats gradient boosting and MLPs on several drugs. The best model changes for 4 of 5 drugs between feature sets, and most gaps (~0.03–0.05 on 93 validation rows) are within noise.
3. **Flexible models are unstable at this sample size.** The MLP is best on Paclitaxel (0.306) but negative on Gefitinib (−0.138). A single decision tree is the weakest model even with depth limits. Ensembles and regularized linear models are the most consistent.
4. **Receptor-level pathways help EGFR inhibitors.** PID wins on 4 of 5 drugs. The biggest gain is on Erlotinib, the hardest drug, where PID adds an explicit EGFR pathway; Gefitinib follows the same trend.
5. **Gene-set choice depends on the drug.** Paclitaxel does better with Hallmark, whose mitotic spindle and G2M sets match its mechanism more directly than PID's signalling-focused sets.
6. **Performance overall is typical; Erlotinib lags.** Peak R² ≈ 0.40 and typical R² of 0.2–0.3 are in line with published transcriptomics-only drug response benchmarks (Kurilov et al., 2020). IC50 assay noise and the lack of mutation data limit the achievable ceiling. Erlotinib, however, is below scores reported in other papers. The likely reasons are that EGFR response depends heavily on mutation status, and that pathway aggregation may dilute single-gene EGFR signal.

## Roadmap

1. **Gene-level features.** Try gene-level feature selection on raw expression, instead of or alongside pathway scores, and compare against the GSVA approach. Start with 5–10 literature mechanism genes per drug (e.g. EGFR/ERBB2/KRAS for Erlotinib/Gefitinib, ESR1/PGR for Tamoxifen, RRM1/RRM2/DCK for Gemcitabine) in ElasticNet, to test whether pathway aggregation is diluting the signal.
2. **Combined pathway set.** HALLMARK_ + PID_ (~246 features), since the two collections complement each other (Paclitaxel vs. Erlotinib).
3. **Final test-set evaluation.** Lock in real numbers on the held-out test split for the winning model per drug, selected on validation only.
4. **SHAP / feature interpretation.** Tie top pathways (or genes, depending on 1) back to each drug's known mechanism, with `PID_EGFR1_PATHWAY` → Erlotinib/Gefitinib as the sanity check.
5. **Robust estimates.** Repeated cross-validation across seeds, to report R² with confidence intervals instead of a single 93-row validation score.
6. **Multi-omic features.** Add mutation status (e.g. EGFR, KRAS), which expression alone captures poorly.

## Usage

GSVA scores are computed in R (`notebooks/gsva.r`) and saved per collection (`data/gsva_<collection>_scores.csv`). Training is in `notebooks/model_training.ipynb`:

```python
from src import data, models, train, results_store

# Join GSVA pathway scores with drug response
df = data.build_pathway_enriched_data(
    required_data_path="data/required_data.parquet",
    gsva_scores_path="data/gsva_C2_scores.csv",
)

# Grouped 70/10/20 split, saved to disk
train_df, val_df, test_df = data.make_splits(df)
data.save_splits(train_df, val_df, test_df, outdir="data/test", suffix="C2")

# Train each stage on one feature set ("HALLMARK_" or "PID_")
results, fitted, run_id = train.run_stage(
    models.baseline_models(), train_df, val_df, stage="baseline", feature_prefix="PID_"
)
results, fitted, run_id = train.run_stage(
    models.candidate_models(), train_df, val_df, stage="candidates", feature_prefix="PID_"
)

# Best model per drug across all logged runs
results_store.best_models(results_store.load_results())
```

## Literature survey

- Fustero-Torre et al. (2022). Pathway-based drug response prediction; the motivation for using pathway enrichment scores over raw genes. [*Nat Commun*](https://www.nature.com/articles/s41467-022-33291-z.pdf)
- Kurilov, Haibe-Kains & Brors (2020). *Assessment of modelling strategies for drug response prediction in cell lines and xenografts.* Scientific Reports. Benchmark for expected R².
- Yang et al. (2013). *Genomics of Drug Sensitivity in Cancer (GDSC).* Nucleic Acids Research.
- Hänzelmann et al. (2013). *GSVA: gene set variation analysis for microarray and RNA-Seq data.* BMC Bioinformatics.
- Liberzon et al. (2015). *The Molecular Signatures Database Hallmark gene set collection.* Cell Systems.
- [RPKM/FPKM/TPM explained](https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/): background on the RNA-seq normalization units used in preprocessing.