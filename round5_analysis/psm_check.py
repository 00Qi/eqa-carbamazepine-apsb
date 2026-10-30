src=open('eqa_revise/eqa-apsb-code/EQA_PSM_Analysis_Final.py').read()
src=src.split('# SECTION 8')[0]
src=src.replace("D:/Dropbox/EQA/EQA_with_LabInfo_integrated.csv","/workspace/eqa_revise/pca_check/EQA_with_LabInfo_integrated.csv").replace("D:/Dropbox/EQA/eqa_ml/EQA_PSM_outputs_final","/workspace/r5/psm_out")
src=src.replace("rcParams['font.family']        = 'Arial'","")
src+="""
print('matched records', len(df_matched), 'yes', len(qa_yes_bias), 'no', len(qa_no_bias))
lm=lab_matched.groupby('QA_bin').mean_abs_bias.agg(['mean','median','count']); print(lm)
from scipy.stats import mannwhitneyu, wilcoxon
a=lab_matched[lab_matched.QA_bin==1].mean_abs_bias; b=lab_matched[lab_matched.QA_bin==0].mean_abs_bias
print('lab-mean MWU', mannwhitneyu(a,b))
m=matched_df.merge(lab_df[['Lab_ID','mean_abs_bias']].rename(columns={'Lab_ID':'treated_lab','mean_abs_bias':'t'})).merge(lab_df[['Lab_ID','mean_abs_bias']].rename(columns={'Lab_ID':'control_lab','mean_abs_bias':'c'}))
print('paired wilcoxon', wilcoxon(m.t,m.c))
g=df.groupby('Lab_ID')
for c in ['QA','SIL-IS','CNAS']:
    print(c,'labs with changing status:', (g[c].nunique()>1).sum())
print('ties exactly 0.5 QA:', (g.QA_bin.mean()==0.5).sum())
print('records per matched lab', df_matched.groupby('Lab_ID').size().describe())
print('smd after', {k:round(v,3) for k,v in smd_after.items()})
print('matched yes median', qa_yes_bias.median(), 'no median', qa_no_bias.median())
"""
exec(compile(src,'psm','exec'))
