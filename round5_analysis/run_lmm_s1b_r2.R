# Round 5b (29 Oct 2026): corrected covariate coding; run from a folder containing EQA_with_LabInfo_integrated.csv
.libPaths(c(path.expand("~/R/lib45"), .libPaths()))
suppressPackageStartupMessages({library(lme4); library(lmerTest); library(broom.mixed); library(MuMIn); library(performance)})
df <- read.csv("EQA_with_LabInfo_integrated.csv", check.names=FALSE, stringsAsFactors=FALSE)
cat("raw rows:", nrow(df), " labs:", length(unique(df$Lab_ID)), " years:", range(df$Year), "\n")
# ---- Corrected encoding (Round 5b; verified against raw Balance_Precision / Separate_Weighing_QC_Sample values) ----
# QC (independent weighing): identical to S&F_0527.R classify_qc -- Yes if the field records a weighing
#   (contains "Yes", a separator or any digit); empty / "Not Provided" / "Not Recorded" / "Not Performed" / "Not Weighed" -> No.
classify_qc <- function(val) {
  if (is.na(val) || val == "") return("No")
  if (grepl("Yes|、|;|\\d", val, ignore.case=TRUE)) return("Yes")
  return("No")
}
# Balance: High = questionnaire code 1 or 2 (2 = 10^-5 g; Round 5c), multi-code answers containing 1/2, or explicit 10^-5/10^-6 g.
#   History: (5b) escaped the unescaped "10^-5|..." regex; (5c) code "2" confirmed High by author.
classify_bal <- function(val) {
  # High = code 1 or 2 (2 = 10^-5 g; Round 5c), multi-option containing 1/2, or explicit 10^-5/10^-6 g
  if (is.na(val) || val == "") return("Normal")
  v <- trimws(as.character(val)); s <- tolower(v)
  if (v %in% c("1","2") || grepl("^1\\s*[,(]", v) || grepl("^2\\s*[\uFF1B;,]", v) ||
      grepl("10\\^-5|10\\^-6|micro|0\\.01 ?mg|0\\.001 ?mg|1 µg|class i", s)) return("High")
  return("Normal")
}
# S&F_0527 originals for comparison
classify_qc0 <- function(val) { if (is.na(val)) return("No"); if (grepl("Yes|、|;|\\d", val, ignore.case=TRUE)) return("Yes"); "No" }
classify_bal0 <- function(val) { if (is.na(val)) return("Normal"); if (val == "1" || grepl("10^-5|10^-6|micro|0.01mg|class i", tolower(val))) return("High"); "Normal" }

df$Abs_Bias <- abs(df$Bias)
cat("NA Bias rows:", sum(is.na(df$Abs_Bias)), "\n"); print(unique(df[is.na(df$Abs_Bias), c("Year","Lab_ID")]))
df <- df[!is.na(df$Abs_Bias), ]
df$QC_Clean <- vapply(df$Separate_Weighing_QC_Sample, classify_qc, character(1))
df$Balance_Clean <- vapply(df$Balance_Precision, classify_bal, character(1))
cat("QC coding agreement with S&F_0527:", mean(df$QC_Clean == vapply(df$Separate_Weighing_QC_Sample, classify_qc0, character(1))), "\n")
cat("Balance coding agreement with S&F_0527:", mean(df$Balance_Clean == vapply(df$Balance_Precision, classify_bal0, character(1))), "\n")
df$Year_Centered <- df$Year - 2014
# Lab_type is blank for 2 laboratory-years (101087/2018, 112013-9/2020); both labs are Hospital/Institute in all other
#   years, so blank -> Hospital here (S&F_0527.R now fills it from the same lab; read_csv NA previously dropped 10 records).
df$Lab_Type_Factor <- factor(ifelse(grepl("CRO", toupper(df$Lab_type)), "CRO", "Hospital"), levels=c("Hospital","CRO"))
df$QA_Factor <- factor(ifelse(df$QA == "Yes", "Yes", "No"), levels=c("No","Yes"))
df$SIL_Factor <- factor(ifelse(df[["SIL-IS"]] == "Yes", "Yes", "No"), levels=c("No","Yes"))
df$QC_Factor <- factor(df$QC_Clean, levels=c("No","Yes"))
df$Balance_Factor <- factor(df$Balance_Clean, levels=c("Normal","High"))
df$Sample_Level <- factor(df$Sample_Level, levels=paste0("Level.",1:5))
tabn <- table(df$Lab_ID); surv <- names(tabn)[tabn >= 30]
df$Cohort_Factor <- factor(ifelse(df$Lab_ID %in% surv, "Survivor", "New_Entrant"), levels=c("New_Entrant","Survivor"))
cat("N=", nrow(df), " labs=", length(unique(df$Lab_ID)), " >100:", sum(df$Abs_Bias>100), " >300:", sum(df$Abs_Bias>300), " max:", max(df$Abs_Bias), "\n")
print(df[df$Abs_Bias>100, c("Year","Lab_ID","Sample_Level","QA","Abs_Bias")])

fit_one <- function(dat, label) {
  m <- lmer(Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor +
              QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID), data=dat, REML=FALSE)
  tid <- broom.mixed::tidy(m, effects="fixed", conf.int=TRUE); qa <- tid[tid$term=="QA_FactorYes", ]
  an <- anova(update(m, . ~ . - QA_Factor:Sample_Level), m)
  r2m <- MuMIn::r.squaredGLMM(m)
  r2p <- performance::r2_nakagawa(m)
  ic <- performance::icc(m)
  vc <- as.data.frame(VarCorr(m)); v_lab <- vc$vcov[vc$grp=="Lab_ID"]; v_res <- vc$vcov[vc$grp=="Residual"]
  out <- data.frame(model=label, n=nrow(dat), n_labs=length(unique(dat$Lab_ID)),
    QA_Level1_est=qa$estimate, QA_Level1_se=qa$std.error, QA_Level1_p=qa$p.value,
    QA_Level1_CI_low=qa$conf.low, QA_Level1_CI_high=qa$conf.high, QA_x_Level_LRT_p=an$`Pr(>Chisq)`[2],
    R2_marginal=r2m[1,"R2m"], R2_conditional=r2m[1,"R2c"],
    R2m_performance=r2p$R2_marginal, R2c_performance=r2p$R2_conditional,
    ICC_adjusted=ic$ICC_adjusted, ICC_unadjusted=ic$ICC_unadjusted,
    var_lab=v_lab, var_resid=v_res, stringsAsFactors=FALSE)
  print(out); out
}
r0 <- fit_one(df, "Primary (untruncated Abs_Bias)")
m0 <- lmer(Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor + QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID), data=df, REML=FALSE)
write.csv(broom.mixed::tidy(m0, effects="fixed", conf.int=TRUE), "Table_S1_primary.csv", row.names=FALSE)
cat("Balance High n:", sum(df$Balance_Clean=="High"), " QC Yes n:", sum(df$QC_Clean=="Yes"), "\n")
d1 <- df; d1$Abs_Bias <- pmin(d1$Abs_Bias, 100); r1 <- fit_one(d1, "Sensitivity: Abs_Bias truncated at 100%")
m1 <- lmer(Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor + QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID), data=d1, REML=FALSE)
write.csv(broom.mixed::tidy(m1, effects="fixed", conf.int=TRUE), "Table_S1b_trunc100_coefs.csv", row.names=FALSE)
d3 <- df; d3$Abs_Bias <- pmin(d3$Abs_Bias, 300); r3 <- fit_one(d3, "Sensitivity: Abs_Bias truncated at 300%")
dx <- df[df$Abs_Bias <= 100, ]; rx <- fit_one(dx, "Sensitivity: exclude Abs_Bias > 100%")
tab <- rbind(r0, r1, r3, rx)
write.csv(tab, "S1b_with_R2_ICC.csv", row.names=FALSE)
# PCA n per year (as in S&F_0527: wide by level, na.omit)
raw <- read.csv("EQA_with_LabInfo_integrated.csv", check.names=FALSE)
for (yr in sort(unique(raw$Year))) { s <- raw[raw$Year==yr,]; w <- reshape(s[,c("Lab_ID","Sample_Level","Bias")], idvar="Lab_ID", timevar="Sample_Level", direction="wide")
  cat(yr, " labs:", nrow(w), " complete:", sum(complete.cases(w)), "\n") }
