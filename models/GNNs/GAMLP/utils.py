
import torch
import numpy as np
import pdb
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support

def label_platten(labels, num_classes):
    # input [b,16]
    y = torch.zeros(size=(labels.shape[0], num_classes), dtype=torch.float32)
    valid_mask = (labels != -1)
    has_zero = torch.any(labels==0, dim=1)
    labels = torch.where(valid_mask, labels, torch.tensor(0))
    y.scatter_(1, labels, 1)
    y[:,0][~has_zero] = 0

    # ouput [b,c]
    return y

def evaluate(logits, y):
    # pdb.set_trace()
    logits = torch.sigmoid(logits).cpu()
    predictions = (logits > 0.5).int()
    targets = y.cpu()
    
    # subset_acc
    subset_acc = (predictions == targets).all(dim=1).float().mean()
    

    # #
    # accuracy = torchmetrics.Accuracy(task='multilabel', num_labels=y.shape[1])
    # acc_value = accuracy(predictions, targets)
    # # print("Accuracy:", acc_value.item())

    # 
    precision = torchmetrics.Precision(task='multilabel', num_labels=y.shape[1])
    precision_value = precision(predictions, targets)
    # print("Precision:", precision_value.item())

    # 
    recall = torchmetrics.Recall(task='multilabel', num_labels=y.shape[1])
    recall_value = recall(predictions, targets)
    # print("Recall:", recall_value.item())

    # 
    f1_value = (2*precision_value*recall_value) / (precision_value+recall_value)
    # f1 = torchmetrics.F1(task='multilabel', num_labels=y.shape[1])
    # f1_value = f1(predictions, targets)
    # print("F1:", f1_value)

    return subset_acc, precision_value, recall_value, f1_value

def average(a,b,c,d,n):
    return a/n, b/n, c/n, d/n



