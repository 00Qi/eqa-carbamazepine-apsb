.libPaths(c(path.expand("~/R/lib45"), .libPaths()))
suppressPackageStartupMessages({
  library(dplyr); library(stringr); library(tidyr); library(ggplot2); library(gridExtra); library(grid)
})
# Round 5c: regenerate Figure 3 with corrected Balance coding (code 2 = 10^-5 High)
df <- read.csv("/workspace/eqa_revise/pca_check/EQA_with_LabInfo_integrated.csv",
               check.names=FALSE, stringsAsFactors=FALSE, na.strings=c("","NA"))
df$Abs_Bias <- abs(df$Bias)
df <- df %>% group_by(Lab_ID) %>% tidyr::fill(Lab_type, .direction="downup") %>% ungroup()

classify_qc <- function(val) {
  if (is.na(val)) return("No")
  if (grepl("Yes|、|;|\\d", val, ignore.case=TRUE)) return("Yes")
  return("No")
}
classify_bal <- function(val) {
  if (is.na(val) || trimws(val) == "") return("Normal")
  v <- trimws(val)
  if (v %in% c("1","2") || str_detect(v, "^1\\s*[,(]") || str_detect(v, "^2\\s*[；;,]") ||
      str_detect(tolower(v), "10\\^-5|10\\^-6|micro|0\\.01 ?mg|0\\.001 ?mg|1 µg|class i")) return("High")
  return("Normal")
}
df <- df %>% mutate(
  QA_Clean = ifelse(str_to_title(QA) == "Yes", "Yes", "No"),
  SIL_Clean = ifelse(str_to_title(`SIL-IS`) == "Yes", "Yes", "No"),
  QC_Clean = vapply(Separate_Weighing_QC_Sample, classify_qc, character(1)),
  Balance_Clean = vapply(Balance_Precision, classify_bal, character(1))
)
cat("Balance High (all rows with Bias):", sum(df$Balance_Clean=="High" & !is.na(df$Abs_Bias)), "\n")

theme_jpa <- function() {
  theme_minimal(base_size = 14) +
    theme(
      text = element_text(family = "sans", color = "black"),
      plot.title = element_text(face = "bold", size = 16, hjust = 0),
      plot.subtitle = element_text(size = 12, color = "#444444", margin = margin(b = 10)),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(color = "black"),
      panel.grid.major = element_line(color = "#e0e0e0", linetype = "dashed"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.ticks.length = unit(0.15, "cm"),
      legend.position = "top",
      legend.justification = "center",
      legend.margin = margin(b = -5),
      strip.text = element_text(face = "bold", size = 12),
      plot.margin = margin(15, 15, 15, 15)
    )
}

# Panel A
df_trends <- df %>%
  filter(Year <= 2020, !is.na(Abs_Bias)) %>%
  group_by(Year) %>%
  summarise(
    QA = mean(QA_Clean == "Yes") * 100,
    SIL = mean(SIL_Clean == "Yes") * 100,
    QC = mean(QC_Clean == "Yes") * 100,
    Bal = mean(Balance_Clean == "High") * 100,
    .groups = "drop"
  )
print(df_trends)
write.csv(df_trends, "/workspace/r5c/fig3_panelA_rates.csv", row.names=FALSE)

df_trends_long <- df_trends %>%
  pivot_longer(cols = -Year, names_to = "Metric", values_to = "Rate") %>%
  mutate(Metric_Label = case_when(
    Metric == "QA" ~ "Independent QA",
    Metric == "SIL" ~ "SIL-IS Usage",
    Metric == "QC" ~ "Separate QC",
    Metric == "Bal" ~ "High-Prec Balance"
  )) %>%
  mutate(Rate = ifelse(Year == 2016 & Metric_Label %in% c("Separate QC", "High-Prec Balance"), NA, Rate))

df_bridge <- df_trends_long %>%
  filter(Metric_Label %in% c("Separate QC", "High-Prec Balance"), Year %in% c(2015, 2017))
df_labels <- df_trends_long %>% filter(Year == 2020)

colors_infra <- c(
  "Independent QA" = "#e74c3c",
  "SIL-IS Usage" = "#3498db",
  "Separate QC" = "#2ecc71",
  "High-Prec Balance" = "#9b59b6"
)

p3a <- ggplot() +
  geom_line(data = df_bridge, aes(x = Year, y = Rate, color = Metric_Label, group = Metric_Label),
            linewidth = 1.0, linetype = "dashed", alpha = 0.6) +
  geom_line(data = df_trends_long %>% filter(!is.na(Rate)),
            aes(x = Year, y = Rate, color = Metric_Label, group = Metric_Label), linewidth = 1.2) +
  geom_point(data = df_trends_long %>% filter(!is.na(Rate)),
             aes(x = Year, y = Rate, color = Metric_Label),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_text(data = df_labels,
            aes(label = Metric_Label, x = 2020.35, y = Rate, color = Metric_Label),
            hjust = 0, fontface = "bold", size = 4, show.legend = FALSE) +
  scale_color_manual(values = colors_infra) +
  scale_x_continuous(breaks = 2014:2020, expand = expansion(mult = c(0.05, 0.35))) +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(title = "Panel A: Infrastructure Modernization", y = "Adoption Rate (%)", x = "Year") +
  theme_jpa() +
  theme(legend.position = "none", plot.margin = margin(r = 10))

# Panel B
prep_box_with_check <- function(col, label, yes="Yes", no="No") {
  sub <- df %>% filter(!is.na(Abs_Bias), !is.na(.data[[col]]), .data[[col]] %in% c(yes, no))
  sub$Status_Type <- ifelse(sub[[col]] == yes, "Yes", "No")
  sub$Status_Type <- factor(sub$Status_Type, levels = c("Yes", "No"))
  n_yes <- sum(sub$Status_Type == "Yes"); n_no <- sum(sub$Status_Type == "No")
  sub <- sub %>% mutate(
    Factor = label,
    Status_Label = ifelse(Status_Type == "Yes", paste0("N=", n_yes), paste0("N=", n_no))
  )
  pval <- wilcox.test(Abs_Bias ~ Status_Type, data=sub)$p.value
  list(data = sub %>% select(Factor, Status_Label, Status_Type, Abs_Bias), p = pval)
}
d1 <- prep_box_with_check("QA_Clean", "1. QA Oversight")
d2 <- prep_box_with_check("SIL_Clean", "2. SIL-IS Usage")
d3 <- prep_box_with_check("QC_Clean", "3. Separate QC")
d4 <- prep_box_with_check("Balance_Clean", "4. High-Prec Balance", "High", "Normal")
df_3b <- bind_rows(d1$data, d2$data, d3$data, d4$data)
df_3b$Status_Type <- factor(df_3b$Status_Type, levels = c("Yes", "No"))
pvals <- c(d1$p, d2$p, d3$p, d4$p)
cat("Panel B p-values:", pvals, "\n")
cat("Balance N High/Normal:", sum(d4$data$Status_Type=="Yes"), sum(d4$data$Status_Type=="No"), "\n")

pos_dodge <- position_dodge(width = 0.8)
p3b <- ggplot(df_3b, aes(x = Factor, y = Abs_Bias, fill = Status_Type)) +
  stat_boxplot(geom = "errorbar", width = 0.3, position = pos_dodge, linewidth = 0.5) +
  geom_boxplot(outlier.shape = NA, width = 0.6, position = pos_dodge, alpha = 1, color = "black", linewidth = 0.4) +
  scale_fill_manual(values = c("Yes" = "#2ecc71", "No" = "#e74c3c"), name="Status", labels=c("Yes", "No")) +
  annotate("label", x = 1:4, y = 21,
           label = ifelse(pvals < 0.001, "P < 0.001", sprintf("P = %.3f", pvals)),
           fill = "#FFF9C4", color = "black", fontface = "bold", size = 3.5,
           label.padding = unit(0.2, "lines"), label.r = unit(0.15, "lines")) +
  geom_text(aes(label=Status_Label, y=0), position = pos_dodge, size=3, fontface="italic", vjust=1.5) +
  coord_cartesian(ylim = c(0, 25)) +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.05))) +
  labs(title = "Panel B: Univariate Performance Gap", y = "Absolute Bias (%)", x = "") +
  theme_jpa() +
  theme(legend.position = "top", legend.title = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"))

p3_final <- arrangeGrob(p3a, p3b, ncol = 2)
ggsave("/workspace/r5c/Figure_3.png", p3_final, width = 16, height = 7, dpi = 600)
ggsave("/workspace/apsb_out/APSB_submission_1008/figures/Figure_3.png", p3_final, width = 16, height = 7, dpi = 600)
cat("Figure 3 saved.\n")
