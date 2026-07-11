library("GSVA")
library(arrow)
library(msigdbr)
library(dplyr)

required_data <- read_parquet("D:/CancerRx/data/required_data.parquet")
meta_cols <-  c("model_id", "model_name", "DRUG_ID", "DRUG_NAME","DATASET",
                    "NLME_RESULT_ID","NLME_CURVE_ID","CELL_LINE_NAME","SANGER_MODEL_ID","CANCER_TYPE",
                  "PUTATIVE_TARGET", "PATHWAY_NAME", "MIN_CONC",
                  "MAX_CONC", "LN_IC50", "AUC", "RMSE", "Z_SCORE")
unique_rows <- required_data |> distinct(model_id, .keep_all = TRUE)
genecols <- setdiff(colnames(unique_rows),meta_cols)

log_tpm_expr <- t(as.matrix(unique_rows[,genecols]))
# drops NaNs
log_tpm_expr <- log_tpm_expr[complete.cases(log_tpm_expr), ]
colnames(log_tpm_expr) <- unique_rows$model_id
mode(log_tpm_expr) <- "numeric"

#hallmark pathways
#hallmark <- msigdbr(species = "Homo sapiens", collection = "H")
pathways <- msigdbr(species = "Homo sapiens", collection = "C2",subcollection = "CP:PID")
gene_sets <- split(pathways$gene_symbol,pathways$gs_name)

# --- GSVA ---
gsva_par <- gsvaParam(log_tpm_expr, gene_sets, kcdf = "Gaussian")
gsva_scores <- gsva(gsva_par)   # gene sets x samples

write.csv(as.data.frame(gsva_scores), "data/gsva_C2_scores.csv")
