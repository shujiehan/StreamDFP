import sys
import math
import pandas as pd
import numpy as np
ls_measure = [
    "classified instances", "classifications correct (percent)",
    "Kappa Statistic (percent)", "Kappa Temporal Statistic (percent)",
    "Kappa M Statistic (percent)", "TP", "FP", "TN", "FN",
    "False Alarm Rate (percent)", "Average Days before Failure",
    "F1 Score (percent)", "F1 Score for class 0 (percent)",
    "F1 Score for class 1 (percent)", "Precision (percent)",
    "Precision for class 0 (percent)", "Precision for class 1 (percent)",
    "Recall (percent)", "Recall for class 0 (percent)",
    "Recall for class 1 (percent)"
]
dict_ = {
    "classified instances": "clf_insts",
    "classifications correct (percent)": "clf_corrct",
    "Kappa Statistic (percent)": "Kappa",
    "Kappa Temporal Statistic (percent)": "Kappa_temp",
    "Kappa M Statistic (percent)": "Kappa_M",
    "TP": "TP",
    "FP": "FP",
    "TN": "TN",
    "FN": "FN",
    "False Alarm Rate (percent)": "FAR",
    "Average Days before Failure": "Days",
    "F1 Score (percent)": "F1_score",
    "F1 Score for class 0 (percent)": "F1_score_c0",
    "F1 Score for class 1 (percent)": "F1_score_c1",
    "Precision (percent)": "Precision",
    "Precision for class 0 (percent)": "Precision_c0",
    "Precision for class 1 (percent)": "Precision_c1",
    "Recall (percent)": "Recall",
    "Recall for class 0 (percent)": "Recall_c0",
    "Recall for class 1 (percent)": "Recall_c1"
}
date = ""
with open(sys.argv[1], "r") as f:
    res = [[]]
    res_row = []
    for line in f.readlines():
        if line[0:3] == "201":
            date = line[0:10]
            if len(res_row) == 41:
                res.append(res_row)
                res_row = []
            res_row.append(date)
        elif "Global Measurements" in line:
            continue
        elif "Local Measurements" in line:
            continue
            #if res is not None:
            #    for item in res:
            #        print(item, end=" ")
            #    print("")
            #flag = 1
            #res = []
        elif "Model measurements" in line:
            continue
        elif "None" in line:
            continue
        elif "time" in line:
            continue
        elif "Index" in line:
            continue
        elif "reset" in line:
            continue
        elif "num drifts" in line:
            continue
        elif "Votes" in line:
            continue
        elif line[0] == "\n":
            continue
        elif "Using" in line:
            continue
        elif "model" in line:
            continue
        elif "tree" in line:
            continue
        elif "leaves" in line:
            continue
        elif "leaf" in line:
            continue
        elif "byte" in line:
            continue
        elif "get" in line:
            continue
        elif "num iterations" in line:
            continue
        elif "prediction time" in line:
            continue
        elif "training time" in line:
            continue
        else:
            res_row.append(line.strip().split(" ")[-1])
res.append(res_row)
columns_name = ['date']
for item in ls_measure:
    columns_name.append("g_%s" % dict_[item])
for item in ls_measure:
    columns_name.append("l_%s" % dict_[item])

df = pd.DataFrame(res, columns=columns_name)
df.to_csv(sys.argv[1][:-4] + ".csv", index=False)
df = df.dropna(how="all", axis=0)
df = df[df['l_Recall_c1'] != "NaN"]
df['l_Days'] = df['l_Days'].astype(np.float32)
df['l_FP'] = df['l_FP'].astype(np.float32)
df['l_FAR'] = df['l_FAR'].astype(np.float32)
df['l_F1_score_c1'] = df['l_F1_score_c1'].astype(np.float32)
df['l_Precision_c1'] = df['l_Precision_c1'].astype(np.float32)
df['l_Recall_c1'] = df['l_Recall_c1'].astype(np.float32)
days_mean = df['l_Days'].mean()
fp_mean = df['l_FP'].mean()
far_mean = df['l_FAR'].mean()
p_mean = df['l_Precision_c1'].mean()
r_mean = df['l_Recall_c1'].mean()
f1_mean = 2.0 * p_mean * r_mean / (p_mean + r_mean)
print("days\t\tFP\t\tFPR\t\tF1-score\tPrecision\tRecall")
print("%lf\t%lf\t%lf\t%lf\t%lf\t%lf" % (days_mean, fp_mean, far_mean, f1_mean,
                                        p_mean, r_mean))
f1 = 2.0 * p_mean * r_mean / (p_mean + r_mean)
