# Cancer Drug Sensitivity Prediction from Gene Expression
### GDSC · ElasticNet · MLP · SHAP · Multi-drug · Reproducible

> **Biological question:** Can a cancer cell line's gene expression profile predict its sensitivity to a drug — and do the predictive genes match known biology?

---

## Results

| Drug | Samples | ElasticNet r | MLP r | Top SHAP gene | Known mechanism? |
|------|---------|-------------|-------|---------------|-----------------|
| Erlotinib | ~300 | — | — | — | EGFR inhibitor ✓ |
| Gefitinib | ~200 | — | — | — | EGFR inhibitor ✓ |
| Paclitaxel | ~700 | — | — | — | Microtubule stabilizer |
| Tamoxifen | ~400 | — | — | — | ER antagonist |
| Gemcitabine | ~600 | — | — | — | Nucleoside analog |

*Fill in after running. Erlotinib is the primary benchmark drug.*

---

## What this is

A end-to-end ML pipeline that predicts how sensitive a cancer cell line is to a drug (measured as IC50) from its gene expression profile alone. Built on the GDSC2 dataset (~970 cell lines, ~17,000 genes, 5 drugs).

**Why this is a biology project, not just an ML project:** SHAP values identify which genes drive predictions for each drug. For Erlotinib (an EGFR inhibitor), EGFR expression should be the top feature — a result independently validated in clinical trials. This kind of model-to-mechanism grounding is the point.

---

## Repo structure

```
gdsc-drug-sensitivity/
│
├── data/
│   ├── raw/                    # Downloaded files go here (gitignored)
│   └── processed/              # Merged, cleaned matrices
│
├── notebooks/
│   ├── 01_eda.ipynb            # Data loading, QC, PCA
│   ├── 02_elasticnet.ipynb     # Baseline model, all drugs
│   ├── 03_mlp.ipynb            # MLP, honest comparison vs baseline
│   └── 04_shap.ipynb           # Feature importance + biology interpretation
│
├── src/
│   ├── data_loader.py          # Download + merge expression & IC50
│   ├── features.py             # Feature selection (top-N variable genes)
│   ├── models.py               # ElasticNet and MLP wrappers
│   ├── evaluate.py             # Pearson r, RMSE, plots
│   └── shap_analysis.py        # SHAP beeswarm + gene lookup
│
├── results/
│   ├── figures/                # PCA plots, SHAP beeswarms, r² comparisons
│   └── metrics.csv             # Pearson r per drug per model
│
├── Makefile                    # Single-command reproducibility
├── requirements.txt
├── environment.yml             # Conda env
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/gdsc-drug-sensitivity
cd gdsc-drug-sensitivity

# Option A: conda (recommended)
conda env create -f environment.yml
conda activate gdsc-ds

# Option B: pip
pip install -r requirements.txt
```

### 2. Download data

```bash
make download
```

This fetches two files automatically:
- `GDSC2_fitted_dose_response_*.xlsx` — IC50 values per cell line per drug
- `Cell_line_RMA_proc_basalExp.txt.zip` — Gene expression (RMA-normalized microarray, ~17k genes)

> Manual alternative: download from [cancerrxgene.org/downloads/bulk_download](https://www.cancerrxgene.org/downloads/bulk_download) and place in `data/raw/`.

### 3. Run full pipeline

```bash
make all
```

This runs in order: data preprocessing → feature selection → ElasticNet → MLP → SHAP → figures.

### 4. Run for a single drug

```bash
make drug DRUG=Erlotinib
```

### 5. View results

```bash
open results/figures/    # SHAP plots, PCA, model comparison
cat results/metrics.csv  # Pearson r per drug per model
```

---

## Pipeline in depth

### Data

**Expression:** RMA-normalized microarray expression for ~970 cancer cell lines (~17,000 genes). Each row is a cell line, each column a gene.

**Drug response:** IC50 values from GDSC2 (log-transformed µM). Lower IC50 = more sensitive to the drug. We filter to drugs with ≥150 cell lines after joining — anything less and train/test splits become unreliable.

**Join:** Cell line names are inconsistent across files. We normalize to COSMIC IDs where available, then fall back on fuzzy name matching. The join logic is in `src/data_loader.py`.

### Feature selection

Raw 17k genes → top 500 most variable genes (by variance across cell lines). This is the feature set used for all models. Rationale: low-variance genes carry no predictive signal and add noise. This is not cheating — it's standard practice and selection is done before the train/test split.

### Models

**ElasticNet** (primary baseline): L1 + L2 regularized linear regression. Hyperparameters (alpha, l1_ratio) tuned via 5-fold CV on training set. Interpretable, fast, and surprisingly competitive.

**MLP** (comparison): 3-layer fully connected network (500 → 256 → 128 → 1), ReLU activations, dropout 0.3, trained with Adam. With ~150–300 samples per drug, expect MLP to match but not consistently beat ElasticNet. Any gap is reported honestly.

**Evaluation:** 80/20 train/test split, stratified by cancer tissue type. Metric: Pearson r (reported in literature) and RMSE. Cross-validation Pearson r reported separately to show variance.

### SHAP interpretation

SHAP (SHapley Additive exPlanations) assigns each gene a contribution score to each prediction. We use `shap.LinearExplainer` for ElasticNet and `shap.DeepExplainer` for MLP.

Output per drug: beeswarm plot of top 20 genes, with SHAP value on x-axis and expression level as color. High expression pushing prediction left (lower IC50) = gene whose expression makes cells more sensitive.

**Biological grounding:** For each top gene, we look up its known role using `mygene.info` API (automated) and provide a one-paragraph interpretation in `notebooks/04_shap.ipynb`.

---

## Biology interpretation (Erlotinib)

Erlotinib is a tyrosine kinase inhibitor that blocks EGFR (Epidermal Growth Factor Receptor). EGFR is a cell-surface receptor that, when overactivated through mutation or overexpression, drives uncontrolled cell proliferation. Erlotinib competitively binds EGFR's ATP-binding pocket and blocks this signal.

**Prediction:** Cell lines with high EGFR expression → low IC50 (sensitive). Cell lines with low/absent EGFR → high IC50 (resistant — the drug has nothing to block).

**What we expect in SHAP:** EGFR near the top, with high expression negatively correlated with IC50. If ERBB2 (HER2, a related receptor) or MET (a bypass resistance gene) also appear, that's biologically consistent. If entirely unrelated genes dominate, something is wrong upstream (data join issue or data leakage).

This is not a post-hoc rationalization — it's a prediction made before running the model, which makes it a genuine validation.

---

## Multi-drug design

Each drug is run as an independent prediction task (separate train/test splits, separate models, separate SHAP). This is deliberate: drug mechanisms are different enough that a shared model would obscure interpretability without substantially improving performance at this sample size.

The five drugs were chosen to span different mechanisms:

| Drug | Target | Signal type |
|------|--------|-------------|
| Erlotinib | EGFR (TKI) | Strong expression signal expected |
| Gefitinib | EGFR (TKI) | Should replicate Erlotinib findings |
| Paclitaxel | Tubulin (mitosis) | Weaker expression signal, broader effect |
| Tamoxifen | Estrogen receptor | Strong signal in ER+ breast cancer lines |
| Gemcitabine | DNA replication | Metabolic genes expected |

Gefitinib is intentionally included as a **replication check**: same target as Erlotinib, different compound. If EGFR appears in both SHAP plots, confidence in the finding increases substantially.

---

## Limitations

- **Cell lines ≠ patients.** GDSC cell lines are grown in controlled conditions. Expression profiles and drug responses can diverge significantly from in vivo tumors. These predictions are not clinical.
- **Microarray, not RNA-seq.** GDSC expression data is RMA-normalized microarray, not sequencing. It is well-validated for this use case but lacks isoform resolution and has a different noise structure.
- **Sample size.** Some drugs have <200 matched cell lines. Results for these drugs are less reliable. We report sample size alongside all metrics.
- **No multi-omics.** Mutation status, copy number variation, and methylation also predict drug sensitivity. Expression alone is not the full picture — it is just the most predictive single modality for pan-cancer analysis.

---

## Environment

```
python=3.11
pandas=2.2
numpy=1.26
scikit-learn=1.4
torch=2.3
shap=0.45
matplotlib=3.8
seaborn=0.13
requests=2.31
openpyxl=3.1       # for reading GDSC xlsx
mygene=3.2         # gene ID/annotation lookup
```

Full pinned environment: `environment.yml`

---

## Reproducing exactly

Every random seed is fixed. `make all` runs the entire pipeline from raw downloaded files to final figures. Intermediate files are cached in `data/processed/` so re-runs are fast.

```bash
# Full clean run from scratch
make clean && make all

# Just re-run SHAP (fast, uses cached model)
make shap DRUG=Erlotinib
```

Expected runtime on a laptop (CPU): ~15 minutes for all 5 drugs end-to-end.

---

## References

1. Iorio et al. (2016). A landscape of pharmacogenomic interactions in cancer. *Cell*, 166(3), 740–754.
2. Yang et al. (2012). Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. *Nucleic Acids Research*, 41(D1), D955–D961.
3. Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

## License

MIT. Data from GDSC is subject to their [terms of use](https://www.cancerrxgene.org/downloads/bulk_download) (non-commercial research only).
