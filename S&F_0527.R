# ==============================================================================
# FINAL PRODUCTION CODE (2024-2025)
# Manuscript: "Operational Rigor as the Determinant of Bioanalytical Quality"
# ==============================================================================
setwd("D:/Dropbox/EQA")
# setwd("C:/Users/qiwen/Dropbox/EQA")
# 1. Environment Setup
# ------------------------------------------------------------------------------
packages <- c("tidyverse", "lme4", "lmerTest", "broom.mixed", "gridExtra", "MuMIn", "effectsize", "ggrepel")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(gridExtra)
library(MuMIn)
library(effectsize)
library(car)

# 2. Output Directory Logic
# ------------------------------------------------------------------------------
# Auto-detect existing folders and increment to avoid overwriting
all_dirs <- list.dirs(path = ".", full.names = FALSE, recursive = FALSE)
fig_dirs <- grep("^Figure_[0-9]+$", all_dirs, value = TRUE)
if (length(fig_dirs) == 0) { run_num <- 1 } else {
  run_num <- max(as.numeric(gsub("Figure_", "", fig_dirs))) + 1
}
output_dir <- paste0("Figure_", run_num)
dir.create(output_dir)
cat(sprintf("\n[INFO] All outputs will be saved to: /%s/\n", output_dir))

# 3. Global Theme (JPA Publication Standard)
# ------------------------------------------------------------------------------
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
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      axis.ticks = element_line(color = "black", size = 0.5),
      axis.ticks.length = unit(0.15, "cm"),
      legend.position = "top", 
      legend.justification = "center",
      legend.margin = margin(b = -5),
      strip.text = element_text(face = "bold", size = 12),
      plot.margin = margin(15, 15, 15, 15)
    )
}

# 4. Data Loading & Strict Cleaning
# ------------------------------------------------------------------------------
# Ensure the CSV is in the working directory
if(!file.exists("EQA_with_LabInfo_integrated.csv")) stop("Data file not found!")
df <- read_csv("EQA_with_LabInfo_integrated.csv", show_col_types = FALSE)
df$Abs_Bias <- abs(df$Bias)

# Cleaning Functions
classify_qc <- function(val) {
  if (is.na(val)) return("No")
  if (grepl("Yes|、|;|\\d", val, ignore.case=TRUE)) return("Yes")
  return("No")
}
classify_bal <- function(val) {
  if (is.na(val)) return("Normal") 
  if (val == "1" || str_detect(tolower(val), "10^-5|10^-6|micro|0.01mg|class i")) return("High")
  return("Normal")
}

# Apply Cleaning
df <- df %>% mutate(
  # Binary Variables
  QA_Clean = ifelse(QA == "Yes", "Yes", "No"),
  SIL_Clean = ifelse(`SIL-IS` == "Yes", "Yes", "No"),
  QC_Clean = sapply(Separate_Weighing_QC_Sample, classify_qc),
  Balance_Clean = sapply(Balance_Precision, classify_bal),
  Lab_Type_Clean = ifelse(str_detect(toupper(Lab_type), "CRO"), "CRO", "Hospital"),
  
  # Factors for LMM (Reference Levels Set)
  Year_Centered = Year - 2014,
  Lab_Type_Factor = factor(Lab_Type_Clean, levels=c("Hospital", "CRO")), # Ref: Hospital
  QA_Factor = factor(QA_Clean, levels=c("No", "Yes")),     # Ref: No
  SIL_Factor = factor(SIL_Clean, levels=c("No", "Yes")),   # Ref: No
  QC_Factor = factor(QC_Clean, levels=c("No", "Yes")),     # Ref: No
  Balance_Factor = factor(Balance_Clean, levels=c("Normal", "High")), # Ref: Normal
  
  # Ensure Level is clean (Level.1 ... Level.5)
  Sample_Level = factor(Sample_Level, levels = c("Level.1", "Level.2", "Level.3", "Level.4", "Level.5"))
)

# Define Cohort (Survivor vs New)
survivor_ids <- df %>% filter(!is.na(Bias)) %>% count(Lab_ID) %>% filter(n >= 30) %>% pull(Lab_ID)
df <- df %>% mutate(
  Cohort_Type = ifelse(Lab_ID %in% survivor_ids, "Survivor", "New_Entrant"),
  Cohort_Factor = factor(Cohort_Type, levels=c("New_Entrant", "Survivor"))
)

# ==============================================================================
# PART A: STATISTICAL ANALYSIS (FINAL INTERACTION MODEL)
# ==============================================================================

# ==============================================================================
# 拟合 Linear Mixed-Effects Model (LMM)
# ==============================================================================
cat("\n--- Fitting LMM with Interactions ---\n")
model_interaction <- lmer(
  Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor +
    QC_Factor + Balance_Factor +         # Control Variables
    QA_Factor * Sample_Level +           # Interaction 1
    SIL_Factor * Sample_Level +          # Interaction 2
    (1|Lab_ID), 
  data = df, REML = FALSE
)

# 1. 计算全局效应量 (Global Effect Size: R-squared)
r2_vals <- r.squaredGLMM(model_interaction)
cat(sprintf("Model Global Effect Size (R2) - Marginal: %.3f, Conditional: %.3f\n", r2_vals[1], r2_vals[2]))

# 2. 提取具有 95% 置信区间的固定效应结果 (新增 conf.int = TRUE)
lmm_res <- broom.mixed::tidy(model_interaction, effects = "fixed", conf.int = TRUE) 

# 3. 计算标准化效应量 (Standardized Beta) 
# 这允许审稿人直接比较 QA 和 SIL-IS 到底哪个效应更"大"
std_effects <- effectsize::standardize_parameters(model_interaction, method = "basic")

# 4. 数据合并与格式化输出 (生成完美的 Publication-ready 表格)
lmm_final_table <- lmm_res %>%
  # 合并标准化效应量
  left_join(as.data.frame(std_effects) %>% select(Parameter, Std_Coefficient), 
            by = c("term" = "Parameter")) %>%
  mutate(
    # 显著性星号
    Significance = case_when(p.value < 0.001 ~ "***", p.value < 0.01 ~ "**", p.value < 0.05 ~ "*", TRUE ~ "ns"),
    
    # 原始效应量 (Unstandardized Effect Size)
    Unstandardized_Estimate = sprintf("%.2f", estimate),
    
    # 95% 置信区间 (95% CI)
    CI_95 = sprintf("[%.2f, %.2f]", conf.low, conf.high),
    
    # 标准化效应量 (Standardized Effect Size)
    Standardized_Beta = sprintf("%.3f", Std_Coefficient),
    
    # P值科学计数法
    P_Value = sprintf("%.3e", p.value)
  ) %>%
  # 重新排列并重命名列，直接迎合审稿人的口味
  select(
    Term = term,
    `Effect Size (Unstandardized Beta)` = Unstandardized_Estimate,
    `95% CI` = CI_95,
    `Standardized Effect Size` = Standardized_Beta,
    Statistic = statistic,
    df = df,
    `P-Value` = P_Value,
    Significance
  )

# 5. 保存带有效应量和CI的最终表格
output_file_name <- file.path(output_dir, "Table_S3_LMM_Interaction_Results_with_EffectSize_CI.csv")
write_csv(lmm_final_table, output_file_name)
cat(sprintf("Success! Detailed LMM results saved to: %s\n", output_file_name))

# 计算线性混合效应模型的 VIF
vif_values <- car::vif(model_interaction)
print(vif_values)

# ==============================================================================
# PART B: MAIN FIGURES (Matched to Manuscript_Corrected_Final)
# ==============================================================================

cat("\n--- Generating Figure 1 (Final Production: Margin Fix + Legend) ---\n")

# 1. Data Preparation
df_metrics <- df %>% 
  filter(Year <= 2020) %>% 
  group_by(Year) %>% 
  summarise(
    NIQR = IQR(Bias, na.rm=TRUE) * 0.7413,      
    MedBias = median(abs(Bias), na.rm=TRUE),    
    .groups = 'drop'
  )

df_metrics_recent <- data.frame(
  Year = c(2021, 2024), 
  NIQR = c(6.60, 6.25),
  MedBias = c(2.8, 2.5)
)
df_f1 <- bind_rows(df_metrics, df_metrics_recent)

# 2. Scaling & Colors
scale_factor <- 2.5
col_prec <- "#00468B" # Blue
col_acc  <- "#ED0000" # Red
bg_color <- "#FADBD8" 

# 3. Plotting
p1_prod <- ggplot(df_f1, aes(x = Year)) +
  
  # --- A. Background ---
  annotate("rect", xmin=2014.5, xmax=2015.5, ymin=-Inf, ymax=Inf, fill=bg_color, alpha=0.6) +
  
  # --- B. Lines & Points ---
  # Precision (Solid Blue)
  geom_line(aes(y = NIQR), color = col_prec, size = 1.2) +
  # Accuracy (Dashed Red)
  geom_line(aes(y = MedBias * scale_factor), color = col_acc, size = 1.2, linetype = "longdash") +
  
  # Points (Standard Years)
  geom_point(data = filter(df_f1, Year <= 2020), aes(y = NIQR), 
             color = col_prec, size = 3.5, shape = 19) +
  geom_point(data = filter(df_f1, Year <= 2020), aes(y = MedBias * scale_factor), 
             color = col_acc, size = 3.5, shape = 17) + 
  
  # Points (2015 Peak)
  geom_point(data = filter(df_f1, Year == 2015), aes(y = NIQR), 
             color = col_prec, size = 6, shape = 1, stroke = 2) +
  
  # Points (2021/2024 New Normal)
  geom_point(data = filter(df_f1, Year >= 2021), aes(y = NIQR), 
             color = col_prec, size = 5, shape = 18) + 
  geom_point(data = filter(df_f1, Year >= 2021), aes(y = NIQR), 
             color = col_prec, size = 7, shape = 5, stroke = 1.2) +
  geom_point(data = filter(df_f1, Year >= 2021), aes(y = MedBias * scale_factor), 
             color = col_acc, size = 5, shape = 18) +
  geom_point(data = filter(df_f1, Year >= 2021), aes(y = MedBias * scale_factor), 
             color = col_acc, size = 7, shape = 5, stroke = 1.2) +
  
  # --- C. Annotations (Events) ---
  annotate("segment", x = 2015, xend = 2015, y = 21, yend = 18.5, 
           color = "black", arrow = arrow(length = unit(0.2, "cm"))) +
  annotate("text", x = 2015, y = 21.5, label = "2015: 7.22 & ChP", 
           color = "black", size = 4, fontface = "bold", vjust = 0) +
  
  geom_vline(xintercept = 2016.5, linetype = "dotted", color = "#117A65", size = 1.2) +
  annotate("text", x = 2016.6, y = 21.5, label = "GQCE", 
           color = "#117A65", size = 4, fontface = "bold", hjust = 0, vjust = 0) +
  
  geom_vline(xintercept = 2019.5, linetype = "dotted", color = "#1F618D",size = 1.2) +
  annotate("text", x = 2019.6, y = 21.5, label = "VBP", 
           color = "#1F618D", size = 4, fontface = "bold", hjust = 0, vjust = 0) +
  
  # --- D. 【关键新增】Legend (Manual Custom Legend) ---
  # Precision Legend (Blue)
  annotate("segment", x = 2021.2, xend = 2022.0, y = 22.5, yend = 22.5, 
           color = col_prec, size = 1.2) +
  annotate("point", x = 2021.6, y = 22.5, color = col_prec, size = 3, shape = 19) +
  annotate("text", x = 2022.2, y = 22.5, label = "Precision (Left Axis)", 
           color = col_prec, fontface = "bold", hjust = 0, size = 3.5) +
  
  # Accuracy Legend (Red)
  annotate("segment", x = 2021.2, xend = 2022.0, y = 21.0, yend = 21.0, 
           color = col_acc, size = 1.2, linetype = "longdash") +
  annotate("point", x = 2021.6, y = 21.0, color = col_acc, size = 3, shape = 17) +
  annotate("text", x = 2022.2, y = 21.0, label = "Accuracy (Right Axis)", 
           color = col_acc, fontface = "bold", hjust = 0, size = 3.5) +
  
  # --- E. Axes & Theme ---
  coord_cartesian(ylim = c(0, 23), xlim = c(2013.5, 2024.5), clip = "off", expand = FALSE) +
  
  scale_x_continuous(breaks = c(2014:2021, 2024)) +
  
  scale_y_continuous(
    name = "Precision (NIQR, %)",
    expand = c(0, 0),
    sec.axis = sec_axis(~ . / scale_factor, name = "Accuracy (Median Abs. Bias, %)")
  ) +
  
  theme_jpa() +
  theme(
    axis.title.y.left = element_text(color = col_prec, face="bold"), # 让轴标题颜色也对应
    axis.text.y.left = element_text(color = col_prec, face="bold"),
    axis.line.y.left = element_line(color = col_prec, size = 1.0),
    
    axis.title.y.right = element_text(color = col_acc, face="bold", angle = 90), # 让轴标题颜色也对应
    axis.text.y.right = element_text(color = col_acc, face="bold"),
    axis.line.y.right = element_line(color = col_acc, size = 1.0),
    
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    
    # Margin Fix
    plot.margin = margin(t = 20, r = 50, b = 10, l = 10, unit = "pt")
  ) +
  
  labs(
    title = "Figure 1. Synchronized Improvement in Bioanalytical Precision and Accuracy",
    x = "Year"
  )

# Save
ggsave(file.path(output_dir, "Figure_1_Final_Legend.png"), p1_prod, width = 10, height = 7, dpi = 600)

# ==============================================================================
# REVISED FIGURE 2: Final Polish (No Outliers, With Caps)
# ==============================================================================

cat("\n--- Generating Figure 2 (Final: No Outliers) ---\n")

# 1. Data Preparation
# -------------------------------------------------
# Panel A Data
df_f2a <- df %>% 
  filter(!is.na(Lab_type)) %>%
  mutate(Type = ifelse(str_detect(Lab_type, "Hospital"), "Hospital/Institution", "CRO")) %>%
  count(Year, Type)

# Panel B Data
df_f2b <- df %>% 
  filter(!is.na(Lab_type)) %>%
  mutate(Type = ifelse(str_detect(Lab_type, "Hospital"), "Hospital/Institution", "CRO"))

# 2. Statistics Calculation
# -------------------------------------------------
pval_type <- wilcox.test(Abs_Bias ~ Type, data = df_f2b)$p.value
p_label <- ifelse(pval_type < 0.001, "P < 0.001", sprintf("P = %.3f", pval_type))

# Sample size labels
n_counts <- df_f2b %>% count(Type)
label_cro <- paste0("CRO\n(N=", n_counts$n[n_counts$Type == "CRO"], ")")
label_hosp <- paste0("Hospital/Institute\n(N=", n_counts$n[n_counts$Type == "Hospital/Institution"], ")")

# 3. Plotting Configuration
# -------------------------------------------------
color_cro <- "#e74c3c"  
color_hosp <- "#2980b9" 
color_bg_p <- "#FFF9C4" 

# Y-axis limit (Ensures space for P-value label)
y_max <- 25 
p_pos <- 23 

# 4. Plotting Panel A (Stacked Bar)
# -------------------------------------------------
p2a <- ggplot(df_f2a, aes(x = Year, y = n, fill = Type)) +
  geom_bar(stat = "identity", position = "stack", width = 0.65, color="black", size=0.4) +
  scale_fill_manual(values = c("CRO" = color_cro, "Hospital/Institution" = color_hosp)) +
  scale_x_continuous(breaks = 2014:2020) +
  labs(
    title = "Panel A: Institutional Composition", 
    y = "Number of Laboratories", 
    x = "Year"
  ) + 
  theme_jpa() +
  theme(
    legend.position = c(0.3, 0.85),
    legend.background = element_rect(fill="white", color="black", size=0.3),
    
    # --- 关键修改：增加图例内边距 ---
    legend.margin = margin(t = 6, r = 10, b = 6, l = 6), 
    
    legend.title = element_blank()
  )

# 5. Plotting Panel B (Boxplot: No Outliers, With Caps)
# -------------------------------------------------
p2b <- ggplot(df_f2b, aes(x = Type, y = Abs_Bias, fill = Type)) +
  # Custom Grid
  geom_hline(yintercept = seq(0, y_max, 5), color = "#eeeeee", size = 0.5) +
  
  # --- 须线顶端的横线 (Error Bar Caps) ---
  stat_boxplot(geom = "errorbar", width = 0.2, size = 0.6, color = "black") +
  
  # --- 箱线图主体 (无散点) ---
  geom_boxplot(
    outlier.shape = NA,  # <--- 关键修改：隐藏离群散点
    width = 0.55, 
    color = "black", 
    size = 0.6,
    alpha = 1
  ) +
  
  # Colors
  scale_fill_manual(values = c("CRO" = color_cro, "Hospital/Institution" = color_hosp)) +
  
  # P-value Label (Yellow Rounded Box)
  annotate("label", x = 1.5, y = p_pos, label = p_label, 
           fill = color_bg_p, color = "black", fontface = "bold", size = 4.5, 
           label.padding = unit(0.3, "lines"), label.r = unit(0.2, "lines")) +
  
  # Scales & Theme
  scale_x_discrete(labels = c("CRO" = label_cro, "Hospital/Institution" = label_hosp)) +
  
  # Y-axis limits
  coord_cartesian(ylim = c(0, y_max)) +
  
  labs(
    title = "Panel B: Performance Comparison", 
    y = "Absolute Bias (%)", 
    x = ""
  ) + 
  theme_jpa() + 
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.text.x = element_text(size = 11, lineheight = 0.9)
  )

# 6. Combine
# -------------------------------------------------
p2_final <- grid.arrange(p2a, p2b, ncol = 2)
ggsave(file.path(output_dir, "Figure 2_Final.png"), p2_final, width = 13, height = 6, dpi = 600)

cat("Figure 2 (No Outliers) saved.\n")

# ==============================================================================
# FIGURE 3: Final Correction (QA/SIL Continuous, QC/Bal Gap)
# ==============================================================================

cat("\n--- Generating Figure 3 (Precise Gap Handling) ---\n")

library(ggrepel)
library(gridExtra)

# 1. Data Preparation (No Global Filter)
# -------------------------------------------------
if(!"QA_Clean" %in% names(df)) {
  df <- df %>% mutate(
    QA_Clean = ifelse(str_to_title(QA) == "Yes", "Yes", "No"),
    SIL_Clean = ifelse(str_to_title(`SIL-IS`) == "Yes", "Yes", "No"),
    QC_Clean = ifelse(str_detect(Separate_Weighing_QC_Sample, "Yes"), "Yes", "No"),
    Balance_Clean = ifelse(str_detect(Balance_Precision, "10^-5"), "High", "Normal")
  )
}

# --- Panel A Data: Trends ---
# Step 1: Calculate ALL years (including 2016)
df_trends <- df %>%
  filter(Year <= 2020) %>%
  group_by(Year) %>%
  summarise(
    QA = mean(QA_Clean == "Yes") * 100,
    SIL = mean(SIL_Clean == "Yes") * 100,
    QC = mean(QC_Clean == "Yes") * 100,
    Bal = mean(Balance_Clean == "High") * 100,
    .groups = 'drop'
  ) %>%
  pivot_longer(cols = -Year, names_to = "Metric", values_to = "Rate") %>%
  mutate(
    Metric_Label = case_when(
      Metric == "QA" ~ "Independent QA",
      Metric == "SIL" ~ "SIL-IS Usage",
      Metric == "QC" ~ "Separate QC",
      Metric == "Bal" ~ "High-Prec Balance"
    )
  )

# Step 2: Manually create gap ONLY for QC and Balance in 2016
# 将 QC 和 Balance 在 2016 年的值设为 NA，这样 geom_line 会自动断开
df_trends <- df_trends %>%
  mutate(Rate = ifelse(Year == 2016 & Metric_Label %in% c("Separate QC", "High-Prec Balance"), NA, Rate))

# Step 3: Create "Bridge Data" for the dashed line (Only for QC and Balance)
# We need points at 2015 and 2017 to draw the connector
df_bridge <- df_trends %>%
  filter(Metric_Label %in% c("Separate QC", "High-Prec Balance")) %>%
  filter(Year %in% c(2015, 2017))

# Data for Labels (2020)
df_labels <- df_trends %>% filter(Year == 2020)

# --- Panel B Data ---
prep_box_with_check <- function(col, label, yes="Yes", no="No") {
  sub <- df %>% filter(!is.na(!!sym(col))) %>% filter(!!sym(col) %in% c(yes, no))
  # Force Factor Order
  sub$Status_Type <- ifelse(sub[[col]] == yes, "Yes", "No")
  sub$Status_Type <- factor(sub$Status_Type, levels = c("Yes", "No")) 
  
  n_yes <- sum(sub$Status_Type == "Yes")
  n_no <- sum(sub$Status_Type == "No")
  sub <- sub %>%
    mutate(
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

# 2. Plotting Panel A (Precise Gap Handling)
# -------------------------------------------------
colors_infra <- c(
  "Independent QA" = "#e74c3c",    
  "SIL-IS Usage" = "#3498db",      
  "Separate QC" = "#2ecc71",       
  "High-Prec Balance" = "#9b59b6"  
)

p3a <- ggplot() +
  
  # Layer 1: Dashed Bridge (Only for QC and Balance, connecting 2015-2017)
  geom_line(data = df_bridge, aes(x = Year, y = Rate, color = Metric_Label, group = Metric_Label), 
            size = 1.0, linetype = "dashed", alpha = 0.6) +
  
  # Layer 2: Main Solid Lines (Since 2016 is NA for QC/Bal, lines will break automatically)
  # For QA/SIL, 2016 is present, so it will be continuous.
  geom_line(data = df_trends %>% filter(!is.na(Rate)), 
            aes(x = Year, y = Rate, color = Metric_Label, group = Metric_Label), 
            size = 1.2) +
  
  # Layer 3: Points (Only plot where Rate is not NA)
  geom_point(data = df_trends %>% filter(!is.na(Rate)), 
             aes(x = Year, y = Rate, color = Metric_Label), 
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  
  # Labels
  geom_text_repel(
    data = df_labels, 
    aes(label = Metric_Label, x = 2020, y = Rate, color = Metric_Label),
    nudge_x = 0.5, direction = "y", fontface = "bold", size = 4,
    segment.size = 0.5, xlim = c(2020.5, 2022)
  ) +
  
  scale_color_manual(values = colors_infra) +
  # Force x-axis to show integer years including 2016
  scale_x_continuous(breaks = 2014:2020, expand = expansion(mult = c(0.05, 0.25))) + 
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  
  labs(
    title = "Panel A: Infrastructure Modernization",
    #    subtitle = "Solid lines: continuous data; Dashed lines: data gap (QC/Balance 2016)",
    y = "Adoption Rate (%)", x = "Year"
  ) +
  theme_jpa() +
  theme(legend.position = "none", plot.margin = margin(r = 10))


# 3. Plotting Panel B (Confirmed Logic)
# -------------------------------------------------
pos_dodge <- position_dodge(width = 0.8)
y_limit_max <- 25 

p3b <- ggplot(df_3b, aes(x = Factor, y = Abs_Bias, fill = Status_Type)) +
  # Caps
  stat_boxplot(geom = "errorbar", width = 0.3, position = pos_dodge, size = 0.5) +
  # Boxplot
  geom_boxplot(outlier.shape = NA, width = 0.6, position = pos_dodge, alpha = 1, color = "black", size = 0.4) +
  
  scale_fill_manual(values = c("Yes" = "#2ecc71", "No" = "#e74c3c"), 
                    name="Status", labels=c("Yes", "No")) +
  
  annotate("label", x = 1:4, y = 21, 
           label = ifelse(pvals < 0.001, "P < 0.001", sprintf("P = %.3f", pvals)), 
           fill = "#FFF9C4", color = "black", fontface = "bold", size = 3.5,
           label.padding = unit(0.2, "lines"), label.r = unit(0.15, "lines")) +
  geom_text(aes(label=Status_Label, y=0), position = pos_dodge, size=3, fontface="italic", vjust=1.5) +
  
  coord_cartesian(ylim = c(0, y_limit_max)) +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.05))) +
  
  labs(
    title = "Panel B: Univariate Performance Gap", 
    #    subtitle = "Comparison of Absolute Bias (Lower is Better)", 
    y = "Absolute Bias (%)", x = ""
  ) + 
  theme_jpa() +
  theme(legend.position = "top", legend.title = element_blank(), axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"))

# 4. Combine
p3_final <- grid.arrange(p3a, p3b, ncol = 2)
ggsave(file.path(output_dir, "Figure_3_Final.png"), p3_final, width = 16, height = 7, dpi = 600)

cat("Figure 3 (Precise Gap Handling) saved.\n")

# ==============================================================================
# REVISED FIGURE 4: Boxplot + Ribbon CI (Fixed Legend Box)
# ==============================================================================

cat("\n--- Generating Figure 4 (Final: Fixed Legend Box) ---\n")

# 1. Aesthetics Setup
# -------------------------------------------------
col_no <- "#e74c3c"   # Red for No QA
col_yes <- "#2980b9"  # Blue for With QA
dodge_width <- 0.75   # Spacing control

p4_final_legend <- ggplot(df, aes(x = Sample_Level, y = Abs_Bias, fill = QA_Clean, color = QA_Clean)) +
  
  # --- Layer 1: Confidence Interval Ribbons ---
  stat_summary(
    fun.data = mean_cl_normal, 
    geom = "ribbon", 
    aes(group = QA_Clean),
    position = position_dodge(width = dodge_width), 
    alpha = 0.15,  
    color = NA     
  ) +
  
  # --- Layer 2: Whiskers Caps ---
  stat_boxplot(
    geom = "errorbar", 
    width = 0.3, 
    position = position_dodge(width = dodge_width),
    color = "black", 
    size = 0.5,
    show.legend = FALSE
  ) +
  
  # --- Layer 3: Boxplots (No Outliers) ---
  geom_boxplot(
    outlier.shape = NA, 
    width = 0.55, 
    position = position_dodge(width = dodge_width),
    alpha = 0.6,   
    color = "black", 
    size = 0.4,
    show.legend = FALSE # Hide boxplot from legend to keep it clean
  ) +
  
  # --- Layer 4: Mean Trend Lines ---
  stat_summary(
    fun = mean, geom = "line", 
    aes(group = QA_Clean),
    position = position_dodge(width = dodge_width), 
    size = 1.2
  ) +
  
  # --- Layer 5: Mean Points ---
  stat_summary(
    fun = mean, geom = "point", 
    aes(group = QA_Clean, fill = QA_Clean),
    shape = 23, size = 3.5, stroke = 1, color = "white",
    position = position_dodge(width = dodge_width),
    show.legend = TRUE # This generates the nice diamond legend
  ) +
  
  # --- Layer 6: Annotations (Moved Right) ---
  annotate("segment", x = 0.8, xend = 1.2, y = 65, yend = 65, 
           color = "black", size = 0.5) + 
  annotate("segment", x = 0.8, xend = 0.8, y = 65, yend = 60, 
           color = "black", size = 0.5) + 
  annotate("segment", x = 1.2, xend = 1.2, y = 65, yend = 35, 
           color = "black", size = 0.5) + 
  
  annotate("label", x = 1.6, y = 72, label = "QA Effect:\n-62.7% Bias\n(P < 0.001)", 
           fill = "white", color = "black", fontface = "bold", size = 3.5, label.size = NA) +
  
  annotate("segment", x = 1.3, xend = 1.1, y = 72, yend = 66, 
           color = "black", size = 0.5, arrow = arrow(length = unit(0.2, "cm"))) +
  
  # --- Aesthetics & Scales ---
  scale_fill_manual(values = c("No" = col_no, "Yes" = col_yes), 
                    labels = c("QA: No", "QA: Yes")) +
  scale_color_manual(values = c("No" = col_no, "Yes" = col_yes), 
                     labels = c("QA: No", "QA: Yes")) +
  
  coord_cartesian(ylim = c(0, 90)) + 
  
  labs(
    title = "Figure 4. QA Oversight Interaction Analysis",
    subtitle = "Grouped boxplots (outliers removed) overlaid with Mean Trend and 95% CI Ribbons.",
    x = "Concentration Level",
    y = "Absolute Bias (%)",
    fill = "QA Oversight",
    color = "QA Oversight"
  ) +
  theme_jpa() +
  
  # --- THEME ADJUSTMENTS FOR LEGEND ---
  theme(
    # Position
    legend.position = c(0.85, 0.85),
    
    # Background Box: Added explicit margin (padding) inside the box
    legend.background = element_rect(fill = "white", color = "black", size = 0.3),
    legend.margin = margin(t = 6, r = 10, b = 6, l = 6), # <--- 关键修改：增加内边距
    
    # Key styling
    legend.key = element_blank(), # Remove grey background behind keys
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    
    # Remove vertical grid
    panel.grid.major.x = element_blank()
  )

# Save
ggsave(file.path(output_dir, "Figure_4_Final.png"), p4_final_legend, width = 11, height = 7, dpi = 600)
cat("Figure 4 (Legend Box Fixed) saved.\n")

# Figure 5: PCA Analysis (Phase Transition)
# -------------------------------------------------
level_map <- c("Level.1"=1, "Level.2"=2, "Level.3"=3, "Level.4"=4, "Level.5"=5)
df$LNum <- level_map[df$Sample_Level]
scree_list <- list(); load_list <- list()
for(yr in sort(unique(df$Year))) {
  dat <- df %>% filter(Year==yr) %>% select(Lab_ID, LNum, Bias) %>% pivot_wider(names_from=LNum, values_from=Bias) %>% na.omit() %>% select(-Lab_ID)
  if(nrow(dat) > 10) { 
    pca <- prcomp(dat, center=TRUE, scale.=TRUE)
    scree_list[[as.character(yr)]] <- summary(pca)$importance[2,1]
    loads <- pca$rotation[,1]; if(sum(loads)<0) loads <- -loads
    for(i in 1:5) load_list[[length(load_list)+1]] <- data.frame(Year=yr, Level=paste0("L", i), Val=loads[i])
  }
}
p5a <- ggplot(data.frame(Year=as.numeric(names(scree_list)), Var=unlist(scree_list)), aes(x=Year, y=Var*100)) +
  geom_bar(stat="identity", fill="#34495e", color="black", width=0.7) +
  geom_line(color="#e74c3c", size=1.2) +
  coord_cartesian(ylim=c(40, 100)) +
  labs(title="Panel A: Decay of Systemic Error", y="PC1 Var (%)", x="") + theme_jpa()
p5b <- ggplot(do.call(rbind, load_list), aes(x=Level, y=factor(Year), fill=Val)) +
  geom_tile(color="black") + scale_fill_distiller(palette="RdBu", direction=-1, limit=c(-0.7, 0.7), name="Loading") +
  scale_y_discrete(limits=rev) + labs(title="Panel B: Error Structure Heatmap", x="Level", y="Year") + theme_jpa()
p5 <- grid.arrange(p5a, p5b, ncol=2)
ggsave(file.path(output_dir, "Figure_5.png"), p5, width = 14, height = 6, dpi = 600)


# ==============================================================================
# PART C: SUPPLEMENTARY FIGURES (S1-S5)
# ==============================================================================

# Figure S1: Baseline Robustness
ps1 <- ggplot(filter(df, Year==2014), aes(x=Bias)) +
  geom_histogram(aes(y=..density..), bins=30, fill="#bdc3c7", color="black") +
  geom_density(color="#c0392b", size=1.5) +
  labs(title = "Figure S1: Baseline Bias Distribution (2014)", y = "Density", x = "Bias (%)") + theme_jpa()
ggsave(file.path(output_dir, "Figure_S1.png"), ps1, width = 8, height = 6)

# Figure S2: Bridging Analysis
niqr_orig <- df %>% filter(Year<=2020) %>% group_by(Year) %>% summarise(Val = IQR(Bias, na.rm=T)) %>% mutate(Method="ISO 13528 A")
meds <- df %>% filter(Year<=2020) %>% group_by(Year, Sample_Level) %>% summarise(M=median(Conc, na.rm=T), .groups='drop')
niqr_new <- df %>% filter(Year<=2020) %>% left_join(meds, by=c("Year","Sample_Level")) %>% 
  mutate(NBias = (Conc - M)/M*100) %>% group_by(Year) %>% summarise(Val = IQR(NBias, na.rm=T)) %>% mutate(Method="Median-Based")
ps2 <- ggplot(bind_rows(niqr_orig, niqr_new), aes(x=Year, y=Val, color=Method, shape=Method)) +
  geom_line(size=1.2) + geom_point(size=3) +
  labs(title="Figure S2: Bridging Analysis", y="NIQR (%)", x="") + theme_jpa()
ggsave(file.path(output_dir, "Figure_S2.png"), ps2, width = 8, height = 6)

# Figure S3: Decoupling Survivor Bias
# -------------------------------------------------
df_surv <- df %>% filter(Cohort_Type == "Survivor") %>% 
  group_by(Year) %>% summarise(NIQR = IQR(Bias, na.rm=TRUE)) %>% mutate(Grp = "Survivor Cohort")
df_tot <- df %>% group_by(Year) %>% summarise(NIQR = IQR(Bias, na.rm=TRUE)) %>% mutate(Grp = "Total Industry")

p3 <- ggplot(bind_rows(df_tot, df_surv), aes(x = Year, y = NIQR, color = Grp, linetype = Grp)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  scale_color_manual(values = c("Total Industry" = "#27ae60", "Survivor Cohort" = "#95a5a6")) +
  labs(title = "Figure S3: Decoupling Survivor Bias", y = "NIQR (%)", x = "") + theme_jpa()
ggsave(file.path(output_dir, "Figure_S3.png"), p3, width = 9, height = 6)

# Figure S4: Youden Plot (2015)
df_2015 <- df %>% filter(Year == 2015, Sample_Level %in% c("Level.2", "Level.3")) %>%
  select(Lab_ID, Sample_Level, Bias) %>% pivot_wider(names_from = Sample_Level, values_from = Bias) %>% na.omit()
ps4 <- ggplot(df_2015, aes(x = Level.2, y = Level.3)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") +
  geom_abline(intercept = 0, slope = 1, color = "#e74c3c", size = 1.2) +
  geom_point(alpha = 0.6, color = "#2c3e50", size = 3) +
  coord_fixed(ratio = 1, xlim = c(-50,50), ylim = c(-50,50)) +
  labs(title = "Figure S4: Youden Plot (2015)", subtitle = "Circular pattern indicates stochastic operational variance", x = "Bias % (L2)", y = "Bias % (L3)") + theme_jpa()
ggsave(file.path(output_dir, "Figure_S4.png"), ps4, width = 7, height = 7)

# Figure S5: SIL-IS INTERACTION PLOT (NEW!)
# -------------------------------------------------
# Shows NO significant divergence (Parallel lines)
ps5 <- ggplot(df, aes(x = Sample_Level, y = Abs_Bias, color = SIL_Clean, group = SIL_Clean)) +
  stat_summary(fun = mean, geom = "point", size = 4) +
  stat_summary(fun = mean, geom = "line", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2, linewidth = 1) +
  scale_color_manual(values = c("No" = "#95a5a6", "Yes" = "#2ecc71"), 
                     labels = c("Analog/No IS", "SIL-IS")) +
  coord_cartesian(ylim = c(0, 80)) + # Same scale as QA plot for comparison
  labs(
    title = "Figure S5. SIL-IS Interaction Analysis",
    subtitle = "Lack of divergence confirms technology alone does not rescue LLOQ performance",
    x = "Concentration Level",
    y = "Absolute Bias (%)",
    color = "Internal Standard"
  ) +
  theme_jpa()
ggsave(file.path(output_dir, "Figure_S5.png"), ps5, width = 9, height = 6)

cat(sprintf("\n[COMPLETE] All figures updated to match Final Manuscript. Output in: /%s/\n", output_dir))


# --- 模型诊断 (Model Diagnostics) ---
# 1. 检查残差正态性 (Q-Q 图)
# 如果点基本落在直线上，说明符合正态假设
qqnorm(resid(model_interaction))
qqline(resid(model_interaction), col = "red")

# 2. 检查同方差性 (残差 vs 拟合值)
# 如果点像一团云一样随机分布（没有明显的喇叭口形状），说明符合同方差假设
plot(fitted(model_interaction), resid(model_interaction), 
     xlab = "Fitted Values", ylab = "Residuals", 
     main = "Residuals vs Fitted")

# ==============================================================================
# MISSING SUPPLEMENTARY TABLES GENERATION
# ==============================================================================
cat("\n--- Generating Supplementary Tables ---\n")

# ==============================================================================
# SUPPLEMENTARY TABLES GENERATION (With ISO 13528 Algorithm A)
# ==============================================================================
cat("\n--- Generating Supplementary Tables (S1 & S2) ---\n")

# ------------------------------------------------------------------------------
# 0. 定义 ISO 13528 Algorithm A 函数 (计算稳健平均值)
# ------------------------------------------------------------------------------
iso13528_algo_a <- function(x, tol = 1e-5, max_iter = 50) {
  x <- na.omit(x)
  if(length(x) == 0) return(NA)
  if(length(x) == 1) return(x)
  
  # 初始化
  x_star <- median(x)
  s_star <- 1.483 * median(abs(x - x_star))
  
  if(s_star == 0) return(x_star) # 避免除零
  
  for(i in 1:max_iter) {
    delta <- 1.5 * s_star
    # 截尾处理 (Winsorization-like clipping)
    x_clip <- pmax(x_star - delta, pmin(x_star + delta, x))
    
    # 更新均值和标准差
    x_star_new <- mean(x_clip)
    s_star_new <- 1.134 * sd(x_clip)
    
    # 检查收敛
    if(abs(x_star_new - x_star) < tol) {
      return(x_star_new)
    }
    
    x_star <- x_star_new
    s_star <- s_star_new
  }
  return(x_star) # 未收敛则返回最后一次结果
}

# ------------------------------------------------------------------------------
# 1. 计算稳健靶值 (Robust Consensus Value)
# ------------------------------------------------------------------------------
if("Conc" %in% names(df)) {
  
  cat("[INFO] Calculating Robust Mean (ISO 13528 Algorithm A) as Assigned Targets...\n")
  
  # 按 Year + Level 分组计算稳健均值
  df_targets <- df %>%
    filter(!is.na(Conc)) %>%
    group_by(Year, Sample_Level) %>%
    summarise(
      Assigned_Conc = iso13528_algo_a(Conc),
      .groups = "drop"
    )
  
  # 将计算出的靶值合并回主数据框
  df_analysis <- df %>%
    left_join(df_targets, by = c("Year", "Sample_Level"))
  
  # ------------------------------------------------------------------------------
  # 2. 生成 Table S1: 线性回归指标
  # ------------------------------------------------------------------------------
  cat("[INFO] Generating Table S1 (Regression Metrics)...\n")
  
  df_reg <- df_analysis %>%
    filter(!is.na(Conc), !is.na(Assigned_Conc)) %>%
    group_by(Year, Lab_ID) %>%
    filter(n() >= 3) %>% 
    summarise(
      Slope = coef(lm(Conc ~ Assigned_Conc))[2],
      Intercept = coef(lm(Conc ~ Assigned_Conc))[1],
      .groups = "drop"
    )
  
  # 汇总统计量
  table_s1 <- df_reg %>%
    group_by(Year) %>%
    summarise(
      Slope_Median = median(Slope, na.rm = TRUE),
      Slope_IQR = IQR(Slope, na.rm = TRUE),
      Intercept_Median = median(Intercept, na.rm = TRUE),
      Intercept_IQR = IQR(Intercept, na.rm = TRUE)
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  write.csv(table_s1, file.path(output_dir, "Table_S1_Regression_Metrics.csv"), row.names = FALSE)
  cat(sprintf("[SUCCESS] Table S1 saved to: %s\n", file.path(output_dir, "Table_S1_Regression_Metrics.csv")))
  
} else {
  cat("[ERROR] Column 'Conc' not found. Cannot calculate targets for Table S1.\n")
}

# ------------------------------------------------------------------------------
# 3. 生成 Table S2: LMM 模型结果 (保持不变)
# ------------------------------------------------------------------------------
cat("[INFO] Generating Table S2 (LMM Model Output)...\n")

if(!exists("model_interaction")) {
  cat("[INFO] 'model_interaction' object not found. Re-fitting LMM model...\n")
  # 确保因子正确
  df$Lab_Type_Factor <- factor(
    ifelse(str_detect(toupper(df$Lab_type), "CRO"), "CRO", "Hospital"),
    levels = c("Hospital", "CRO")
  )
  df$QA_Factor <- factor(ifelse(df$QA_Clean=="Yes", "Yes", "No"), levels=c("No", "Yes"))
  df$SIL_Factor <- factor(ifelse(df$SIL_Clean=="Yes", "Yes", "No"), levels=c("No", "Yes"))
  df$QC_Factor <- factor(ifelse(df$QC_Clean=="Yes", "Yes", "No"), levels=c("No", "Yes"))
  df$Balance_Factor <- factor(ifelse(df$Balance_Clean=="High", "High", "Normal"), levels=c("Normal", "High"))
  df$Cohort_Factor <- factor(ifelse(df$Cohort_Type=="Survivor", "Survivor", "New_Entrant"),
                             levels=c("New_Entrant", "Survivor"))
  
  model_interaction <- lmer(Abs_Bias ~ Year_Centered + Lab_Type_Factor + Cohort_Factor +
                              QA_Factor * Sample_Level +
                              SIL_Factor * Sample_Level +
                              QC_Factor + Balance_Factor +
                              (1 | Lab_ID), 
                            data = df, REML = FALSE)
}

table_s2 <- broom.mixed::tidy(model_interaction, effects = "fixed", conf.int = TRUE) %>%
  mutate(
    p.value = format.pval(p.value, digits = 4, eps = 0.001),
    estimate = round(estimate, 3),
    std.error = round(std.error, 3),
    statistic = round(statistic, 3),
    conf.low = round(conf.low, 3),
    conf.high = round(conf.high, 3)
  ) %>%
  select(term, estimate, std.error, statistic, df, p.value, conf.low, conf.high)

write.csv(table_s2, file.path(output_dir, "Table_S2_LMM_Output.csv"), row.names = FALSE)
cat(sprintf("[SUCCESS] Table S2 saved to: %s\n", file.path(output_dir, "Table_S2_LMM_Output.csv")))

# 清理冗余文件
f_s3 <- file.path(output_dir, "Table_S3_LMM_Interaction_Results.csv")
if(file.exists(f_s3)) file.remove(f_s3)

cat("\n[ALL DONE] Supplementary Tables generated with Robust Mean (Algorithm A).\n")

# ==============================================================================
# PART D: SUPPLEMENTARY ANALYSES FOR MANUSCRIPT TEXT REPRODUCIBILITY
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 主成分分析 (PCA) 中的质控组 "质心距离" 计算
# 对应手稿描述："在 PCA 得分空间中，有独立 QA 和无独立 QA 实验室之间的质心距离 (centroid distance) 
# 在 2016-2017 年间达到大于 1.5 的峰值，到 2020 年显著缩小至小于 0.5"
# ------------------------------------------------------------------------------
cat("\n--- Running PCA Centroid Distance Analysis (QA vs Non-QA) ---\n")

pca_distance_list <- list()

for(yr in 2014:2020) {
  # 将数据转换为宽格式：用5个浓度水平的Bias作为PCA特征
  dat_pca <- df %>%
    filter(Year == yr) %>%
    select(Lab_ID, Sample_Level, Bias, QA_Clean) %>%
    pivot_wider(names_from = Sample_Level, values_from = Bias) %>%
    na.omit() # 去除有缺失值的实验室，保证 PCA 正常运行
  
  # 确保样本量足够且同时包含有QA和无QA的实验室
  if(nrow(dat_pca) > 10 && length(unique(dat_pca$QA_Clean)) == 2) {
    
    # 运行 PCA 分析
    pca_res <- prcomp(dat_pca %>% select(starts_with("Level")), center = TRUE, scale. = TRUE)
    
    # 提取前两个主成分得分 (PC1, PC2)
    scores <- as.data.frame(pca_res$x[, 1:2])
    scores$QA <- dat_pca$QA_Clean
    
    # 计算有 QA 和无 QA 两组在 PCA 空间中的质心坐标
    centroids <- scores %>%
      group_by(QA) %>%
      summarise(mean_PC1 = mean(PC1), mean_PC2 = mean(PC2), .groups = 'drop')
    
    # 计算两质心之间的欧氏距离
    if(nrow(centroids) == 2) {
      dist_val <- sqrt(
        (centroids$mean_PC1[centroids$QA == "Yes"] - centroids$mean_PC1[centroids$QA == "No"])^2 +
          (centroids$mean_PC2[centroids$QA == "Yes"] - centroids$mean_PC2[centroids$QA == "No"])^2
      )
      pca_distance_list[[as.character(yr)]] <- data.frame(Year = yr, Centroid_Distance = dist_val)
    }
  }
}

df_pca_dist <- do.call(rbind, pca_distance_list)
cat("\n[RESULT] PCA Centroid Distances by Year:\n")
print(df_pca_dist %>% mutate(Centroid_Distance = round(Centroid_Distance, 3)))

# 导出质心距离数据以支持正文数据点
write_csv(df_pca_dist, file.path(output_dir, "Text_Support_PCA_Centroid_Distances.csv"))
