suppressPackageStartupMessages({
  library(dplyr); library(ggplot2)
})
df <- read.csv('/workspace/eqa_revise/pca_check/EQA_with_LabInfo_integrated.csv',
               check.names=FALSE, stringsAsFactors=FALSE)
cat('cols:', paste(names(df), collapse=', '), '\n')
df$QA_bin <- as.integer(df$QA == 'Yes')
df$Lab_type_bin <- as.integer(grepl('CRO', toupper(trimws(as.character(df$Lab_type)))))
bal_map <- c('1'=1, '2'=2, '1,2'=1, '10^-6 Precision'=3,
             'Model not found; possibly 10^-5 precision'=2,
             '1 (Explicitly Stated)'=1)
df$Bal_enc <- bal_map[trimws(as.character(df$`Balance_Precision`))]
df$Bal_enc[is.na(df$Bal_enc)] <- 1
df$CNAS_bin_rec <- as.integer(df$CNAS == 'Yes')
df$SIL_bin_rec <- as.integer(df$`SIL-IS` == 'Yes')

lab <- df %>% group_by(Lab_ID) %>% summarise(
  QA_bin = as.integer(mean(QA_bin) > 0.5),
  Lab_type_bin = first(Lab_type_bin),
  Years_exp = n_distinct(Year),
  First_year = min(Year),
  CNAS_bin = as.integer(mean(CNAS_bin_rec) > 0.5),
  SIL_IS_bin = as.integer(mean(SIL_bin_rec) > 0.5),
  Bal_enc = median(Bal_enc, na.rm=TRUE),
  Region = first(Region),
  .groups='drop'
)
top_reg <- names(sort(table(df$Region), decreasing=TRUE))[1:5]
lab$Region_enc <- ifelse(lab$Region %in% top_reg, match(lab$Region, top_reg)-1, length(top_reg))

# Propensity score via glm
covs <- c('Lab_type_bin','Years_exp','First_year','CNAS_bin','SIL_IS_bin','Bal_enc','Region_enc')
lab_sc <- lab
for (c in covs) lab_sc[[c]] <- as.numeric(scale(lab[[c]]))
ps_fit <- glm(QA_bin ~ Lab_type_bin + Years_exp + First_year + CNAS_bin + SIL_IS_bin + Bal_enc + Region_enc,
              data=lab_sc, family=binomial())
lab$ps <- predict(ps_fit, type='response')
lab$logit_ps <- qlogis(pmin(pmax(lab$ps, 1e-6), 1-1e-6))
caliper <- 0.2 * sd(lab$logit_ps)
cat('caliper=', round(caliper,4), ' labs=', nrow(lab),
    ' QA=', sum(lab$QA_bin), ' no=', sum(lab$QA_bin==0), '\n')

treated <- lab[lab$QA_bin==1, ]
control <- lab[lab$QA_bin==0, ]
used <- character(0)
pairs <- list()
for (i in seq_len(nrow(treated))) {
  d <- abs(control$logit_ps - treated$logit_ps[i])
  ord <- order(d)
  for (j in ord) {
    cid <- control$Lab_ID[j]
    if (!(cid %in% used) && d[j] <= caliper) {
      used <- c(used, cid)
      pairs[[length(pairs)+1]] <- c(treated$Lab_ID[i], cid)
      break
    }
  }
}
cat('matched pairs=', length(pairs), '\n')
matched_labs <- unique(unlist(pairs))
lab_m <- lab[lab$Lab_ID %in% matched_labs, ]

smd_fun <- function(x1, x2) {
  abs((mean(x1)-mean(x2)) / sqrt((sd(x1)^2 + sd(x2)^2)/2 + 1e-10))
}
labels <- c('Lab type (CRO)','Years of PT participation','First year of participation',
            'CNAS accreditation','SIL-IS status','Balance precision','Region')
smd_b <- smd_a <- numeric(length(covs))
for (k in seq_along(covs)) {
  c <- covs[k]
  smd_b[k] <- smd_fun(lab$QA_bin==1 %>% {lab[[c]][.]} %>% {lab[lab$QA_bin==1, c]},
                      lab[lab$QA_bin==0, c])
  # fix: cleaner
}
# rewrite loop cleanly
for (k in seq_along(covs)) {
  c <- covs[k]
  smd_b[k] <- smd_fun(lab[lab$QA_bin==1, c], lab[lab$QA_bin==0, c])
  smd_a[k] <- smd_fun(lab_m[lab_m$QA_bin==1, c], lab_m[lab_m$QA_bin==0, c])
  cat(sprintf('%-32s before=%.3f after=%.3f\n', labels[k], smd_b[k], smd_a[k]))
}

plot_df <- data.frame(
  covariate = factor(labels, levels=rev(labels)),
  Before = smd_b, After = smd_a
)
plot_long <- tidyr::pivot_longer(plot_df, cols=c(Before, After),
                                 names_to='Stage', values_to='SMD')
plot_long$Stage <- factor(plot_long$Stage, levels=c('Before','After'))

# theme_jpa-ish
theme_jpa <- function() {
  theme_bw(base_size=11) +
    theme(
      plot.title = element_text(face='bold', size=12, hjust=0),
      axis.title = element_text(face='bold', size=11),
      axis.text = element_text(color='black', size=10),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(linestyle='dashed', colour='#e0e0e0', linewidth=0.4),
      legend.position = 'bottom',
      legend.title = element_blank()
    )
}
# linestyle may not work — use linetype
theme_jpa <- function() {
  theme_bw(base_size=11) +
    theme(
      plot.title = element_text(face='bold', size=12, hjust=0),
      axis.title = element_text(face='bold', size=11),
      axis.text = element_text(color='black', size=10),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(linetype='dashed', colour='#e0e0e0', linewidth=0.4),
      legend.position = 'bottom',
      legend.title = element_blank(),
      panel.border = element_rect(colour='black', fill=NA, linewidth=1)
    )
}

seg <- plot_df
p <- ggplot() +
  geom_vline(xintercept=0.10, linetype='dashed', colour='black', linewidth=0.6) +
  geom_segment(data=seg, aes(x=Before, xend=After, y=covariate, yend=covariate),
               colour='#b0b0b0', linewidth=0.8) +
  geom_point(data=plot_long, aes(x=SMD, y=covariate, colour=Stage, shape=Stage), size=3.2) +
  scale_colour_manual(values=c(Before='#ED0000', After='#00468B'),
                      labels=c('Before matching','After matching')) +
  scale_shape_manual(values=c(Before=16, After=18),
                     labels=c('Before matching','After matching')) +
  labs(title='Figure S1. PSM covariate balance (Love plot)',
       x='Absolute standardised mean difference (SMD)', y=NULL) +
  theme_jpa() +
  xlim(0, max(c(smd_b, smd_a, 0.5)) + 0.12)

out <- '/workspace/r5d/psm_out/Figure_S1_Love.png'
ggsave(out, p, width=8, height=5.2, dpi=300, bg='white')
ggsave('/workspace/r5d/psm_out/Figure_S1_Love.tiff', p, width=8, height=5.2, dpi=300, bg='white')
write.csv(data.frame(covariate=labels, SMD_before=round(smd_b,3), SMD_after=round(smd_a,3),
                     n_pairs=length(pairs)),
          '/workspace/r5d/psm_out/psm_smd_round5.csv', row.names=FALSE)
cat('saved', out, '\n')
