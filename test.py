TP = 134657
TN = 19748
FP = 16086
FN = 8816

PR = TP / (TP + FP)
RC = TP / (TP + FN)
print(RC)
print(PR)
print(2 * (PR * RC) / (PR + RC))
