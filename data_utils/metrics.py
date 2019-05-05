# Copyright (c) Microsoft. All rights reserved.
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr

def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, [p[1] for p in predicts])[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, [p[1] for p in predicts])[0]
    return 100.0 * scof

def compute_rocauc(predicts, labels):
    return 100.0 * roc_auc_score(labels, [p[1] for p in predicts])

def compute_prauc(predicts, labels):
    precision, recall, thresholds = precision_recall_curve(labels, [p[1] for p in predicts])
    prauc = auc(recall, precision)
    return 100.0 * prauc
