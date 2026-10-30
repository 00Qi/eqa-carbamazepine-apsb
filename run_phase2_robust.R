# Phase-2 robust LMM re-estimation (2026-10-30)
# Same covariate structure as Round 5c; primary outcome log1p(Abs_Bias)
.libPaths(c(path.expand("~/R/lib45"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(broom.mixed); library(MuMIn); library(performance)
})
set.seed(20261030)
outdir <- "/workspace/p2/out"
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

df0 <- read.csv("EQA_with_LabInfo_integrated.csv", check.names=FALSE, stringsAsFactors=FALSE)
cat("RAW rows:", nrow(df0), " labs:", length(unique(df0$Lab_ID)),
    " years:", paste(range(df0$Year), collapse="-"), "\n")

# ---- encoding (Round 5c) ----
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

# Data-flow by year
flow <- aggregate(Lab_ID ~ Year, data=df0, FUN=function(x) length(unique(x)))
names(flow)[2] <- "n_labs"
flow$n_records <- as.integer(table(df0$Year)[as.character(flow$Year)])
flow$n_bias_missing <- sapply(flow$Year, function(y) sum(is.na(df0$Bias[df0$Year==y])))
write.csv(flow, file.path(outdir, "Table_S0_NCCL_year_flow.csv"), row.names=FALSE)
cat("=== NCCL year flow ===\n"); print(flow)
cat("Total records:", nrow(df0), " unique labs:", length(unique(df0$Lab_ID)), "\n")

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

cat("Modeled N=", nrow(df), " labs=", length(unique(df$Lab_ID)),
    " High bal=", sum(df$Balance_Clean=="High"),
    " >100=", sum(df$Abs_Bias>100), " max=", max(df$Abs_Bias), "\n")

form <- Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor +
  QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID)
form_log <- log1p_Abs ~ Year_Centered + Lab_Type_Factor + Cohort_Factor + QC_Factor + Balance_Factor +
  QA_Factor * Sample_Level + SIL_Factor * Sample_Level + (1|Lab_ID)

fit_summary <- function(m, dat, label, outcome="Abs_Bias") {
  tid <- broom.mixed::tidy(m, effects="fixed", conf.int=TRUE)
  qa <- tid[tid$term=="QA_FactorYes", ]
  an <- anova(update(m, . ~ . - QA_Factor:Sample_Level), m)
  r2m <- tryCatch(MuMIn::r.squaredGLMM(m), error=function(e) matrix(c(NA,NA),1,2, dimnames=list(NULL,c("R2m","R2c"))))
  ic <- tryCatch(performance::icc(m), error=function(e) list(ICC_adjusted=NA))
  vc <- as.data.frame(VarCorr(m))
  v_lab <- vc$vcov[vc$grp=="Lab_ID"]; v_res <- vc$vcov[vc$grp=="Residual"]
  data.frame(model=label, outcome=outcome, n=nrow(dat), n_labs=length(unique(dat$Lab_ID)),
    QA_Level1_est=qa$estimate, QA_Level1_se=qa$std.error, QA_Level1_p=qa$p.value,
    QA_Level1_CI_low=qa$conf.low, QA_Level1_CI_high=qa$conf.high,
    QA_x_Level_LRT_p=an$`Pr(>Chisq)`[2],
    R2_marginal=r2m[1,"R2m"], R2_conditional=r2m[1,"R2c"],
    ICC_adjusted=ic$ICC_adjusted, var_lab=v_lab, var_resid=v_res,
    stringsAsFactors=FALSE)
}

# ---- Primary: untruncated Abs_Bias (reference) ----
m0 <- lmer(form, data=df, REML=FALSE)
r0 <- fit_summary(m0, df, "Primary untruncated Abs_Bias (pp)", "Abs_Bias")
write.csv(broom.mixed::tidy(m0, effects="fixed", conf.int=TRUE),
          file.path(outdir, "Table_S1_primary_pp.csv"), row.names=FALSE)

# ---- New primary: log1p(Abs_Bias) ----
m_log <- lmer(form_log, data=df, REML=FALSE)
r_log <- fit_summary(m_log, df, "PRIMARY log1p(Abs_Bias)", "log1p_Abs")
write.csv(broom.mixed::tidy(m_log, effects="fixed", conf.int=TRUE),
          file.path(outdir, "Table_S1_primary_log1p.csv"), row.names=FALSE)

# Back-transform Level-1 QA contrast
# On log1p scale: E[log1p(Y)|QA=Yes] - E[log1p(Y)|QA=No] = beta
# Multiplicative on (1+AbsBias): ratio = exp(beta)
# Approx pp at mean Level-1 AbsBias among No-QA: (1+mu)*exp(beta) - (1+mu) = (1+mu)*(exp(beta)-1)
l1 <- df[df$Sample_Level=="Level.1", ]
mu_all <- mean(l1$Abs_Bias)
mu_no  <- mean(l1$Abs_Bias[l1$QA_Factor=="No"])
mu_yes <- mean(l1$Abs_Bias[l1$QA_Factor=="Yes"])
med_all <- median(l1$Abs_Bias)
beta <- r_log$QA_Level1_est
ci_lo <- r_log$QA_Level1_CI_low
ci_hi <- r_log$QA_Level1_CI_high
bt <- data.frame(
  beta_log1p=beta, CI_low=ci_lo, CI_high=ci_hi, p=r_log$QA_Level1_p,
  ratio_1plusAbsBias=exp(beta),
  ratio_CI_low=exp(ci_lo), ratio_CI_high=exp(ci_hi),
  approx_pp_at_mean_L1_all=(1+mu_all)*(exp(beta)-1),
  approx_pp_at_mean_L1_NoQA=(1+mu_no)*(exp(beta)-1),
  approx_pp_CI_low_NoQA=(1+mu_no)*(exp(ci_lo)-1),
  approx_pp_CI_high_NoQA=(1+mu_no)*(exp(ci_hi)-1),
  approx_pp_at_median_L1=(1+med_all)*(exp(beta)-1),
  mean_AbsBias_L1_all=mu_all, mean_AbsBias_L1_NoQA=mu_no, mean_AbsBias_L1_Yes=mu_yes,
  median_AbsBias_L1=med_all, n_L1=nrow(l1)
)
write.csv(bt, file.path(outdir, "log1p_backtransform_Level1_QA.csv"), row.names=FALSE)
cat("=== log1p Level-1 QA ===\n"); print(bt)

# ---- Sensitivities: capping (winsorization) and exclusion ----
d_cap100 <- df; d_cap100$Abs_Bias <- pmin(d_cap100$Abs_Bias, 100)
m_c100 <- lmer(form, data=d_cap100, REML=FALSE)
r_c100 <- fit_summary(m_c100, d_cap100, "Sensitivity: Abs_Bias capped@100% (winsor)", "Abs_Bias_capped100")

d_cap300 <- df; d_cap300$Abs_Bias <- pmin(d_cap300$Abs_Bias, 300)
m_c300 <- lmer(form, data=d_cap300, REML=FALSE)
r_c300 <- fit_summary(m_c300, d_cap300, "Sensitivity: Abs_Bias capped@300% (winsor)", "Abs_Bias_capped300")

d_ex <- df[df$Abs_Bias <= 100, ]
m_ex <- lmer(form, data=d_ex, REML=FALSE)
r_ex <- fit_summary(m_ex, d_ex, "Sensitivity: exclude Abs_Bias > 100%", "Abs_Bias_exclude100")

# Also log1p on capped@100 for consistency check
d_cap100$log1p_Abs <- log1p(d_cap100$Abs_Bias)
m_log_c100 <- lmer(form_log, data=d_cap100, REML=FALSE)
r_log_c100 <- fit_summary(m_log_c100, d_cap100, "log1p on capped@100 Abs_Bias", "log1p_capped100")

tab <- rbind(r_log, r0, r_c100, r_c300, r_ex, r_log_c100)
write.csv(tab, file.path(outdir, "S1b_phase2_with_log1p.csv"), row.names=FALSE)
write.csv(broom.mixed::tidy(m_c100, effects="fixed", conf.int=TRUE),
          file.path(outdir, "Table_S1b_capped100_coefs.csv"), row.names=FALSE)
cat("=== S1b summary ===\n"); print(tab[,c("model","QA_Level1_est","QA_Level1_CI_low","QA_Level1_CI_high","QA_Level1_p","QA_x_Level_LRT_p")])

# ---- Lab-cluster bootstrap for log1p primary ----
labs <- unique(df$Lab_ID)
B <- 500
boot_beta <- numeric(B)
boot_ok <- 0
t0 <- Sys.time()
for (b in seq_len(B)) {
  samp <- sample(labs, length(labs), replace=TRUE)
  # rebuild by stacking sampled labs (with new ids if duplicated)
  pieces <- vector("list", length(samp))
  for (i in seq_along(samp)) {
    sub <- df[df$Lab_ID == samp[i], ]
    sub$Lab_ID_boot <- paste0(samp[i], "_", i)
    pieces[[i]] <- sub
  }
  boot_df <- do.call(rbind, pieces)
  boot_df$Lab_ID <- boot_df$Lab_ID_boot
  fitb <- tryCatch(
    lmer(form_log, data=boot_df, REML=FALSE),
    error=function(e) NULL, warning=function(w) invokeRestart("muffleWarning")
  )
  if (is.null(fitb)) next
  fe <- fixef(fitb)
  if (!("QA_FactorYes" %in% names(fe))) next
  boot_beta[b] <- fe[["QA_FactorYes"]]
  boot_ok <- boot_ok + 1
  if (b %% 50 == 0) cat("bootstrap", b, "/", B, " ok=", boot_ok, " elapsed=", round(difftime(Sys.time(),t0,units="mins"),1), "min\n")
}
boot_beta <- boot_beta[seq_len(boot_ok)]
boot_ci <- quantile(boot_beta, c(0.025, 0.975), na.rm=TRUE)
boot_out <- data.frame(
  B_requested=B, B_converged=boot_ok,
  beta_point=beta,
  boot_mean=mean(boot_beta), boot_median=median(boot_beta), boot_se=sd(boot_beta),
  boot_CI_low=boot_ci[1], boot_CI_high=boot_ci[2],
  ratio_point=exp(beta), ratio_boot_CI_low=exp(boot_ci[1]), ratio_boot_CI_high=exp(boot_ci[2]),
  approx_pp_NoQA_point=(1+mu_no)*(exp(beta)-1),
  approx_pp_NoQA_boot_CI_low=(1+mu_no)*(exp(boot_ci[1])-1),
  approx_pp_NoQA_boot_CI_high=(1+mu_no)*(exp(boot_ci[2])-1)
)
write.csv(boot_out, file.path(outdir, "log1p_lab_cluster_bootstrap.csv"), row.names=FALSE)
write.csv(data.frame(boot_beta=boot_beta), file.path(outdir, "log1p_bootstrap_draws.csv"), row.names=FALSE)
cat("=== Bootstrap CI ===\n"); print(boot_out)

# ---- Leave-one-lab-out influence on Level-1 QA (log1p) ----
# Focus on labs with extreme Level-1 Abs_Bias
l1_ext <- l1[order(-l1$Abs_Bias), c("Lab_ID","Year","Abs_Bias","QA")]
write.csv(head(l1_ext, 20), file.path(outdir, "top20_L1_AbsBias.csv"), row.names=FALSE)
# Also labs contributing to >100% records
hi_labs <- unique(df$Lab_ID[df$Abs_Bias > 100])
# LOLO for: all hi_labs + top 10 L1 extremes + random 10 others for context
focus <- unique(c(as.character(hi_labs), as.character(head(l1_ext$Lab_ID, 15))))
cat("LOLO focus labs:", length(focus), " hi>100 labs:", length(hi_labs), "\n")
lolo <- data.frame(dropped_lab=character(), beta=numeric(), CI_low=numeric(), CI_high=numeric(), p=numeric(), stringsAsFactors=FALSE)
for (lab in focus) {
  dsub <- df[df$Lab_ID != lab, ]
  mf <- tryCatch(lmer(form_log, data=dsub, REML=FALSE), error=function(e) NULL)
  if (is.null(mf)) next
  tid <- broom.mixed::tidy(mf, effects="fixed", conf.int=TRUE)
  qa <- tid[tid$term=="QA_FactorYes", ]
  lolo <- rbind(lolo, data.frame(dropped_lab=lab, beta=qa$estimate, CI_low=qa$conf.low, CI_high=qa$conf.high, p=qa$p.value,
                                 stringsAsFactors=FALSE))
}
lolo$delta_beta <- lolo$beta - beta
lolo <- lolo[order(abs(lolo$delta_beta), decreasing=TRUE), ]
write.csv(lolo, file.path(outdir, "leave_one_lab_out_log1p.csv"), row.names=FALSE)
cat("=== Top influence (LOLO) ===\n"); print(head(lolo, 10))

# Full LOLO for all 184 is expensive (~few min); do a sample of remaining if time — skip full, report focus set
# Quick: also drop ALL >100% labs at once
d_nohi <- df[!(df$Lab_ID %in% hi_labs), ]
m_nohi <- lmer(form_log, data=d_nohi, REML=FALSE)
r_nohi <- fit_summary(m_nohi, d_nohi, "log1p drop all labs with any AbsBias>100", "log1p_drop_hi_labs")
write.csv(r_nohi, file.path(outdir, "log1p_drop_hi_labs.csv"), row.names=FALSE)
cat("=== Drop all >100 labs ===\n"); print(r_nohi)

# ---- M0–M4 nested incremental R² (on log1p) ----
# M0: Year + Level + (1|Lab)
# M1: + Lab_Type + Cohort
# M2: M1 + SIL + Balance  (tech block; SIL×Level)
# M3: M1 + QA + QC        (process block; QA×Level)
# M4: full
mM0 <- lmer(log1p_Abs ~ Year_Centered + Sample_Level + (1|Lab_ID), data=df, REML=FALSE)
mM1 <- lmer(log1p_Abs ~ Year_Centered + Sample_Level + Lab_Type_Factor + Cohort_Factor + (1|Lab_ID), data=df, REML=FALSE)
mM2 <- lmer(log1p_Abs ~ Year_Centered + Sample_Level + Lab_Type_Factor + Cohort_Factor +
              SIL_Factor * Sample_Level + Balance_Factor + (1|Lab_ID), data=df, REML=FALSE)
mM3 <- lmer(log1p_Abs ~ Year_Centered + Sample_Level + Lab_Type_Factor + Cohort_Factor +
              QA_Factor * Sample_Level + QC_Factor + (1|Lab_ID), data=df, REML=FALSE)
mM4 <- m_log
r2row <- function(m, label) {
  r2 <- MuMIn::r.squaredGLMM(m)
  data.frame(model=label, R2m=r2[1,"R2m"], R2c=r2[1,"R2c"], AIC=AIC(m), BIC=BIC(m), logLik=as.numeric(logLik(m)))
}
nest <- rbind(
  r2row(mM0, "M0 year+level"),
  r2row(mM1, "M1 +lab type+cohort"),
  r2row(mM2, "M2 tech (SIL×Level+Balance)"),
  r2row(mM3, "M3 process (QA×Level+QC)"),
  r2row(mM4, "M4 joint")
)
nest$delta_R2m_vs_M1 <- nest$R2m - nest$R2m[nest$model=="M1 +lab type+cohort"]
# LR tests vs M1 where nested
lr_M2 <- anova(mM1, mM2)
lr_M3 <- anova(mM1, mM3)
lr_M4_vs_M2 <- anova(mM2, mM4)
lr_M4_vs_M3 <- anova(mM3, mM4)
nest$LR_p_vs_M1 <- c(NA, NA, lr_M2$`Pr(>Chisq)`[2], lr_M3$`Pr(>Chisq)`[2], NA)
write.csv(nest, file.path(outdir, "M0_M4_incremental_R2.csv"), row.names=FALSE)
cat("=== M0-M4 ===\n"); print(nest)
cat("LR M4 vs M2 (add process): p=", lr_M4_vs_M2$`Pr(>Chisq)`[2], "\n")
cat("LR M4 vs M3 (add tech): p=", lr_M4_vs_M3$`Pr(>Chisq)`[2], "\n")
write.csv(data.frame(
  contrast=c("M2_vs_M1_tech","M3_vs_M1_process","M4_vs_M2_add_process","M4_vs_M3_add_tech"),
  LR_p=c(lr_M2$`Pr(>Chisq)`[2], lr_M3$`Pr(>Chisq)`[2], lr_M4_vs_M2$`Pr(>Chisq)`[2], lr_M4_vs_M3$`Pr(>Chisq)`[2]),
  Chisq=c(lr_M2$Chisq[2], lr_M3$Chisq[2], lr_M4_vs_M2$Chisq[2], lr_M4_vs_M3$Chisq[2]),
  Df=c(lr_M2$Df[2], lr_M3$Df[2], lr_M4_vs_M2$Df[2], lr_M4_vs_M3$Df[2])
), file.path(outdir, "M0_M4_LR_tests.csv"), row.names=FALSE)

# ---- Common-mode bias variance decomposition (exploratory support) ----
# For each year: lab-level bias vector b_il; alpha_i = mean across levels; delta_il = b_il - alpha_i
# Var(common)=var(alpha); mean Var(specific)=mean over labs of var(delta)
cm <- data.frame(Year=integer(), n_labs=integer(), var_common=numeric(), mean_var_specific=numeric(),
                 ratio_common_over_total=numeric(), PC1_var_share_ref=numeric(), stringsAsFactors=FALSE)
for (yr in sort(unique(df0$Year))) {
  s <- df0[df0$Year==yr & !is.na(df0$Bias), ]
  w <- reshape(s[,c("Lab_ID","Sample_Level","Bias")], idvar="Lab_ID", timevar="Sample_Level", direction="wide")
  mat <- as.matrix(w[, grep("^Bias\\.", names(w))])
  ok <- complete.cases(mat)
  mat <- mat[ok, , drop=FALSE]
  if (nrow(mat) < 5) next
  alpha <- rowMeans(mat)
  delta <- mat - alpha
  var_c <- var(alpha)
  var_s <- mean(apply(delta, 1, var))
  # total per-observation approx
  tot <- var_c + var_s
  cm <- rbind(cm, data.frame(Year=yr, n_labs=nrow(mat), var_common=var_c, mean_var_specific=var_s,
                             ratio_common_over_total=var_c/tot, PC1_var_share_ref=NA))
}
write.csv(cm, file.path(outdir, "common_mode_bias_decomposition.csv"), row.names=FALSE)
cat("=== Common-mode variance ===\n"); print(cm)

# Decision helper
cat("\n===== NARRATIVE DECISION HELPER =====\n")
cat("log1p QA L1 beta=", beta, " Wald CI=[", ci_lo, ",", ci_hi, "] p=", r_log$QA_Level1_p, "\n")
cat("bootstrap CI=[", boot_ci[1], ",", boot_ci[2], "] n_boot=", boot_ok, "\n")
cat("ratio(1+AbsBias)=", exp(beta), " approx pp at mean NoQA L1=", (1+mu_no)*(exp(beta)-1), "\n")
cat("capped@100 pp=", r_c100$QA_Level1_est, " p=", r_c100$QA_Level1_p, "\n")
cat("exclude>100 pp=", r_ex$QA_Level1_est, " p=", r_ex$QA_Level1_p, "\n")
cat("drop hi-labs log1p beta=", r_nohi$QA_Level1_est, " p=", r_nohi$QA_Level1_p, "\n")
sig_log <- r_log$QA_Level1_p < 0.05 && boot_ci[2] < 0
sig_cap <- r_c100$QA_Level1_p < 0.05
dir_ok <- beta < 0 && r_c100$QA_Level1_est < 0
cat("KEEP_QA_THREAD?", sig_log && dir_ok, " (log1p sig + boot CI<0 + direction; cap100 sig=", sig_cap, ")\n")
cat("DONE\n")
