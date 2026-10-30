.libPaths(c(path.expand("~/R/library"), .libPaths()))
library(lme4)
library(lmerTest)
library(broom.mixed)

df <- read.csv("/workspace/eqa_revise/pca_check/EQA_with_LabInfo_integrated.csv", check.names=FALSE, stringsAsFactors=FALSE)

classify_qc <- function(val) {
  if (is.na(val) || val == "") return("No")
  s <- tolower(trimws(as.character(val)))
  if (s %in% c("yes","y")) return("Yes")
  if (s %in% c("no","n","not provided","not recorded")) return("No")
  return("Yes")
}
classify_bal <- function(val) {
  if (is.na(val) || val == "") return("Normal")
  s <- tolower(as.character(val))
  if (as.character(val) == "1" || grepl("10\\^-5|10\\^-6|micro|0\\.01mg|class i|1 µg|0\\.001 mg", s, ignore.case=TRUE) || as.character(val) == "1,2") return("High")
  return("Normal")
}

df$Abs_Bias <- abs(df$Bias)
df <- df[!is.na(df$Abs_Bias), ]

df$QA_Clean <- ifelse(df$QA == "Yes", "Yes", "No")
df$SIL_Clean <- ifelse(df[["SIL-IS"]] == "Yes", "Yes", "No")
df$QC_Clean <- vapply(df$Separate_Weighing_QC_Sample, classify_qc, character(1))
df$Balance_Clean <- vapply(df$Balance_Precision, classify_bal, character(1))
df$Lab_Type_Clean <- ifelse(grepl("CRO", toupper(df$Lab_type)), "CRO", "Hospital")

df$Year_Centered <- df$Year - 2014
df$Lab_Type_Factor <- factor(df$Lab_Type_Clean, levels=c("Hospital","CRO"))
df$QA_Factor <- factor(df$QA_Clean, levels=c("No","Yes"))
df$SIL_Factor <- factor(df$SIL_Clean, levels=c("No","Yes"))
df$QC_Factor <- factor(df$QC_Clean, levels=c("No","Yes"))
df$Balance_Factor <- factor(df$Balance_Clean, levels=c("Normal","High"))
df$Sample_Level <- factor(df$Sample_Level, levels=c("Level.1","Level.2","Level.3","Level.4","Level.5"))

tabn <- table(df$Lab_ID)
survivor_ids <- names(tabn)[tabn >= 30]
df$Cohort_Factor <- factor(ifelse(df$Lab_ID %in% survivor_ids, "Survivor", "New_Entrant"),
                           levels=c("New_Entrant","Survivor"))

cat("N=", nrow(df), " labs=", length(unique(df$Lab_ID)), "\n", sep="")
cat("Abs_Bias>100:", sum(df$Abs_Bias>100), " >300:", sum(df$Abs_Bias>300), "\n")

fit_one <- function(dat, label) {
  m <- lmer(Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor +
              QC_Factor + Balance_Factor +
              QA_Factor * Sample_Level +
              SIL_Factor * Sample_Level +
              (1|Lab_ID),
            data=dat, REML=FALSE)
  tid <- broom.mixed::tidy(m, effects="fixed", conf.int=TRUE)
  qa <- tid[tid$term=="QA_FactorYes", ]
  m_red <- update(m, . ~ . - QA_Factor:Sample_Level)
  an <- anova(m_red, m)
  inter_p <- an$`Pr(>Chisq)`[2]
  cat(sprintf("\n=== %s N=%d ===\n", label, nrow(dat)))
  cat(sprintf("QA Level1: est=%.4f SE=%.4f CI=[%.4f,%.4f] p=%.6g\n",
              qa$estimate, qa$std.error, qa$conf.low, qa$conf.high, qa$p.value))
  cat(sprintf("QA x Level LRT p=%.6g\n", inter_p))
  list(model=m, tidy=tid, qa=qa, inter_p=inter_p, n=nrow(dat), label=label)
}

r0 <- fit_one(df, "Primary (untruncated Abs_Bias)")

df100 <- df; df100$Abs_Bias <- pmin(df100$Abs_Bias, 100)
r100 <- fit_one(df100, "Sensitivity: Abs_Bias truncated at 100%")

df300 <- df; df300$Abs_Bias <- pmin(df300$Abs_Bias, 300)
r300 <- fit_one(df300, "Sensitivity: Abs_Bias truncated at 300%")

df_ex <- df[df$Abs_Bias <= 100, ]
r_ex <- fit_one(df_ex, "Sensitivity: exclude Abs_Bias > 100%")

outdir <- "/workspace/apsb_out/APSB_submission_1008/supplementary"
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

summarize_row <- function(r) {
  data.frame(
    model=r$label,
    n=r$n,
    QA_Level1_est=r$qa$estimate,
    QA_Level1_se=r$qa$std.error,
    QA_Level1_p=r$qa$p.value,
    QA_Level1_CI_low=r$qa$conf.low,
    QA_Level1_CI_high=r$qa$conf.high,
    QA_x_Level_LRT_p=r$inter_p,
    stringsAsFactors=FALSE
  )
}

tab <- rbind(summarize_row(r0), summarize_row(r100), summarize_row(r300), summarize_row(r_ex))
write.csv(tab, file.path(outdir, "Table_S1b_LMM_truncation_sensitivity.csv"), row.names=FALSE)
write.csv(r0$tidy, file.path(outdir, "Table_S1_LMM_full_coefficients_primary.csv"), row.names=FALSE)
write.csv(r100$tidy, file.path(outdir, "Table_S1b_LMM_coefficients_truncate100.csv"), row.names=FALSE)

fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3g", p))
lines <- c(
  "# Table S1b. LMM sensitivity: truncation / exclusion of extreme absolute % bias",
  "",
  "Primary model (Methods / `S&F_0527.R`):",
  "`Abs_Bias ~ Year_Centered + Lab_Type + Cohort + QC + Balance + QA × Level + SIL-IS × Level + (1|Lab_ID)`, REML = FALSE (`lme4`/`lmerTest`).",
  "",
  "QA Level-1 contrast = fixed-effect coefficient `QA_FactorYes` with Level.1 as the reference concentration.",
  "QA × Level P = likelihood-ratio test vs the nested model without the QA × Level interaction.",
  "",
  "| Model | N | QA Level-1 (pp) | 95% CI | P | QA×Level LRT P |",
  "|---|---:|---:|---|---:|---:|"
)
for (i in seq_len(nrow(tab))) {
  lines <- c(lines, sprintf(
    "| %s | %d | %.2f | %.2f to %.2f | %s | %s |",
    tab$model[i], tab$n[i], tab$QA_Level1_est[i],
    tab$QA_Level1_CI_low[i], tab$QA_Level1_CI_high[i],
    fmt_p(tab$QA_Level1_p[i]), fmt_p(tab$QA_x_Level_LRT_p[i])
  ))
}
lines <- c(lines, "",
  "Only 8/3190 records had Abs_Bias > 100% (4 > 300%). Truncation at 100% attenuates the Level-1 QA point estimate but the QA × Level interaction and a negative Level-1 QA contrast remain.",
  "")
writeLines(lines, file.path(outdir, "Table_S1b_LMM_truncation_sensitivity.md"))
print(tab)
cat("\nWrote Table S1b files to", outdir, "\n")
