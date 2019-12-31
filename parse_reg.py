import sys
import math
import pandas as pd
import numpy as np
ls_measure = [
    "classified instances", "mean absolute error", "root mean squared error",
    "relative mean absolute error", "relative root mean squared error",
    "mean absolute error for failures", "root mean squared error for failures",
    "average error days before failures", "num failures"
]
dict_ = {
    "classified instances": "clf_insts",
    "mean absolute error": "mean_abs_err",
    "root mean squared error": "sqr_mean_err",
    "relative mean absolute error": "rel_mean_abs_err",
    "relative root mean squared error": "rel_sqr_mean_err",
    "mean absolute error for failures": "mean_err_failed",
    "root mean squared error for failures": "sqr_err_failed",
    "average error days before failures": "avg_err_days",
    "num failures": "TP"
}
date = ""
with open(sys.argv[1], "r") as f:
    res = [[]]
    res_row = []
    for line in f.readlines():
        if line[0:3] == "201":
            date = line[0:10]
            if len(res_row) == 19:
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
        elif "option nodes" in line:
            continue
        elif "subtrees" in line:
            continue
        elif "no serial" in line:
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
df['l_avg_err_days'] = df['l_avg_err_days'].astype(np.float32)
days_mean = df['l_avg_err_days'].mean()
days_std = df['l_avg_err_days'].std()
days_max = df['l_avg_err_days'].max()
days_min = df['l_avg_err_days'].min()
print("days_mean\tdays_std\tdays_max\tdays_min")
print("%lf\t%lf\t%lf\t%lf" % (days_mean, days_std, days_max, days_min))
