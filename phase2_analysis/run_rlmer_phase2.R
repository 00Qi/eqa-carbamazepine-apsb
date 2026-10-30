# Phase-2 rlmer robust LMM (robustlmm) — 2026-10-30
.libPaths("/home/box/micromamba/envs/rlm/lib/R/library")
# Same Round5/phase2 covariate structure; Abs_Bias and log1p; capped@100 fallback
suppressPackageStartupMessages({
  library(lme4)
  library(robustlmm)
})
set.seed(20261030)
outdir <- "/workspace/p2/out"
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)
logf <- file.path(outdir, "run_rlmer_log.txt")
sink(logf, split=TRUE)

classify_qc <- function(val) {
  if (is.na(val) || val == "") return("No")
  if (grepl("Yes|、|;|\\d", val, ignore.case=TRUE)) return("Yes")
  "No"
}
classify_bal <- function(val) {
  if (is.na(val) || val == "") return("Normal")
  v <- trimws(as.character(val)); s <- tolower(v)
  if (v %in% c("1","2") || grepl("^1\\s*[,(]", v) || grepl("^2\\s*[\uFF1B;,]", v) ||
      grepl("10\\^-5|10\\^-6|micro|0\\.01 ?mg|0\\.001 ?mg|1 µg|class i", s)) return("High")
  "Normal"
}

df0 <- read.csv("/workspace/p2/analysis/EQA_with_LabInfo_integrated.csv",
                check.names=FALSE, stringsAsFactors=FALSE)
df <- df0
df$Abs_Bias <- abs(df$Bias)
df <- df[!is.na(df$Abs_Bias), ]
df$QC_Clean <- vapply(df$Separate_Weighing_QC_Sample, classify_qc, character(1))
df$Balance_Clean <- vapply(df$Balance_Precision, classify_bal, character(1))
df$Year_Centered <- df$Year - 2014
df$Lab_Type_Factor <- factor(ifelse(grepl("CRO", toupper(df$Lab_type)), "CRO", "Hospital"), levels=c("Hospital","CRO"))
df$QA_Factor <- factor(ifelse(df$QA == "Yes", "Yes", "No"), levels=c("No","Yes"))
df$SIL_Factor <- factor(ifelse(df[["SIL-IS"]] == "Yes", "Yes", "No"), levels=c("No","Yes"))
df$QC_Factor <- factor(df$QC_Clean, levels=c("No","Yes"))
df$Balance_Factor <- factor(df$Balance_Clean, levels=c("Normal","High"))
df$Sample_Level <- factor(df$Sample_Level, levels=paste0("Level.",1:5))
tabn <- table(df$Lab_ID); surv <- names(tabn)[tabn >= 30]
df$Cohort_Factor <- factor(ifelse(df$Lab_ID %in% surv, "Survivor", "New_Entrant"), levels=c("New_Entrant","Survivor"))
df$log1p_Abs <- log1p(df$Abs_Bias)

form <- Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor +
  QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID)
form_log <- log1p_Abs ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor +
  QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID)

extract_qa <- function(m, label, outcome, n, n_labs) {
  fe <- fixef(m)
  # try asymptotic SE / CI from vcov
  V <- tryCatch(as.matrix(vcov(m)), error=function(e) NULL)
  se <- if (!is.null(V) && "QA_FactorYes" %in% rownames(V)) sqrt(V["QA_FactorYes","QA_FactorYes"]) else NA_real_
  est <- unname(fe[["QA_FactorYes"]])
  ci_lo <- if (is.finite(se)) est - 1.96*se else NA_real_
  ci_hi <- if (is.finite(se)) est + 1.96*se else NA_real_
  data.frame(model=label, outcome=outcome, n=n, n_labs=n_labs,
             QA_Level1_est=est, QA_Level1_se=se,
             QA_Level1_CI_low=ci_lo, QA_Level1_CI_high=ci_hi,
             converged=TRUE, note="", stringsAsFactors=FALSE)
}

results <- list()
errors <- list()

# 1) rlmer on Abs_Bias
cat("=== rlmer Abs_Bias ===\n")
t0 <- Sys.time()
tryCatch({
  m_abs <- rlmer(form, data=df)
  cat("elapsed min:", round(as.numeric(difftime(Sys.time(), t0, units="mins")), 2), "\n")
  print(summary(m_abs))
  results[["rlmer_AbsBias"]] <- extract_qa(m_abs, "rlmer Abs_Bias", "Abs_Bias", nrow(df), length(unique(df$Lab_ID)))
  write.csv(data.frame(term=names(fixef(m_abs)), estimate=as.numeric(fixef(m_abs))),
            file.path(outdir, "Table_S1b_rlmer_AbsBias_coefs.csv"), row.names=FALSE)
}, error=function(e) {
  errors[["rlmer_AbsBias"]] <<- conditionMessage(e)
  cat("FAIL rlmer Abs_Bias:", conditionMessage(e), "\n")
  results[["rlmer_AbsBias"]] <<- data.frame(model="rlmer Abs_Bias", outcome="Abs_Bias",
    n=nrow(df), n_labs=length(unique(df$Lab_ID)),
    QA_Level1_est=NA, QA_Level1_se=NA, QA_Level1_CI_low=NA, QA_Level1_CI_high=NA,
    converged=FALSE, note=conditionMessage(e), stringsAsFactors=FALSE)
})

# 2) rlmer on log1p
cat("=== rlmer log1p(Abs_Bias) ===\n")
t0 <- Sys.time()
tryCatch({
  m_log <- rlmer(form_log, data=df)
  cat("elapsed min:", round(as.numeric(difftime(Sys.time(), t0, units="mins")), 2), "\n")
  print(summary(m_log))
  results[["rlmer_log1p"]] <- extract_qa(m_log, "rlmer log1p(Abs_Bias)", "log1p_Abs", nrow(df), length(unique(df$Lab_ID)))
  write.csv(data.frame(term=names(fixef(m_log)), estimate=as.numeric(fixef(m_log))),
            file.path(outdir, "Table_S1b_rlmer_log1p_coefs.csv"), row.names=FALSE)
}, error=function(e) {
  errors[["rlmer_log1p"]] <<- conditionMessage(e)
  cat("FAIL rlmer log1p:", conditionMessage(e), "\n")
  results[["rlmer_log1p"]] <<- data.frame(model="rlmer log1p(Abs_Bias)", outcome="log1p_Abs",
    n=nrow(df), n_labs=length(unique(df$Lab_ID)),
    QA_Level1_est=NA, QA_Level1_se=NA, QA_Level1_CI_low=NA, QA_Level1_CI_high=NA,
    converged=FALSE, note=conditionMessage(e), stringsAsFactors=FALSE)
})

# 3) Fallback: Abs_Bias capped@100
cat("=== rlmer Abs_Bias capped@100 ===\n")
d_cap <- df; d_cap$Abs_Bias <- pmin(d_cap$Abs_Bias, 100)
t0 <- Sys.time()
tryCatch({
  m_cap <- rlmer(form, data=d_cap)
  cat("elapsed min:", round(as.numeric(difftime(Sys.time(), t0, units="mins")), 2), "\n")
  print(summary(m_cap))
  results[["rlmer_capped100"]] <- extract_qa(m_cap, "rlmer Abs_Bias capped@100", "Abs_Bias_capped100",
                                             nrow(d_cap), length(unique(d_cap$Lab_ID)))
  write.csv(data.frame(term=names(fixef(m_cap)), estimate=as.numeric(fixef(m_cap))),
            file.path(outdir, "Table_S1b_rlmer_capped100_coefs.csv"), row.names=FALSE)
}, error=function(e) {
  errors[["rlmer_capped100"]] <<- conditionMessage(e)
  cat("FAIL rlmer capped@100:", conditionMessage(e), "\n")
  results[["rlmer_capped100"]] <<- data.frame(model="rlmer Abs_Bias capped@100", outcome="Abs_Bias_capped100",
    n=nrow(d_cap), n_labs=length(unique(d_cap$Lab_ID)),
    QA_Level1_est=NA, QA_Level1_se=NA, QA_Level1_CI_low=NA, QA_Level1_CI_high=NA,
    converged=FALSE, note=conditionMessage(e), stringsAsFactors=FALSE)
})

tab <- do.call(rbind, results)
write.csv(tab, file.path(outdir, "S1b_rlmer_summary.csv"), row.names=FALSE)
cat("=== SUMMARY ===\n"); print(tab)
if (length(errors)) { cat("=== ERRORS ===\n"); print(errors) }
sink()
cat("Wrote", file.path(outdir, "S1b_rlmer_summary.csv"), "\n")
