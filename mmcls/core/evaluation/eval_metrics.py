# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch
from math import sqrt

from scipy.special import ndtri
from sklearn.metrics import confusion_matrix as cfm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """
    low = []
    high = []
    for i in range(len(r)):
#        print('r: ',r[i])
#        print('n: ',n[i])
        A = 2*r[i] + z**2
        B = z*sqrt(z**2 + 4*r[i]*(1 - r[i]/n[i]))
        C = 2*(n[i] + z**2)
        l = (A-B)/C
        h = (A+B)/C
#        print('l: ',l)
#        print('h: ',h)
        low.append(l)
        high.append(h)
    low_avg = np.array(low).mean()
    high_avg = np.array(high).mean()
    return np.array((low_avg,high_avg))

def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    specificitys = []
    recall_CIs = []
    specificity_CIs = []
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    class_names = ['BENIGN','MALIGNANT','NORMAL']
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        confuse_matrix = cfm(target,_pred_label)
        for title, normalize in titles_options:
            if normalize == None:
                ax = sns.heatmap(confuse_matrix, annot=True, cmap='Blues')
                ax.set_title('Confusion matrix without normalization\n\n');
                ax.set_xlabel('\nPredicted label')
                ax.set_ylabel('True label');

                ## Ticket labels - List must be in alphabetical order
                ax.xaxis.set_ticklabels(class_names)
                ax.yaxis.set_ticklabels(class_names)

                ## Display the visualization of the Confusion Matrix.
#                plt.show()
            else:
                ax = sns.heatmap(confuse_matrix/np.sum(confuse_matrix), annot=True,cmap='Blues',
                                 fmt='.2f',vmin=0,vmax=1)
                ax.set_title('Confusion matrix with normalization\n\n');
                ax.set_xlabel('\nPredicted label')
                ax.set_ylabel('True label');

                ## Ticket labels - List must be in alphabetical order
                ax.xaxis.set_ticklabels(class_names)
                ax.yaxis.set_ticklabels(class_names)

                ## Display the visualization of the Confusion Matrix.
#                plt.show()
        print(confuse_matrix)
        FP = confuse_matrix.sum(axis=0) - np.diag(confuse_matrix)
        FN = confuse_matrix.sum(axis=1) - np.diag(confuse_matrix)
        TP = np.diag(confuse_matrix)
        TN = confuse_matrix.sum() - (FP+FN+TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        TNR = TN/(TN+FP)
#        print('pred: ',_pred_label)
#        print('target: ',target)
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
#        print('precision:',precision)
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        NPV = TN/(TN+FN)
        kap = cohen_kappa_score(target,_pred_label)
        alpha = 0.95
        z = -ndtri((1.0-alpha)/2)
        precision_CI = _proportion_confidence_interval(TP, TP + FP ,z)
        accuracy_CI =  _proportion_confidence_interval(TP+TN, TP + FP + TN + FN ,z)
        recall_CI = _proportion_confidence_interval(TP, TP + FN ,z)
        specificity_CI = _proportion_confidence_interval(TN, TN + FP,z)
        NPV_CI = _proportion_confidence_interval(TN, TN + FN,z)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
            specificity = float(TNR.mean())
            NPV = float(NPV.mean())
        print('specificity : ',specificity)
        print('NPV : ',NPV)
        print('accuracy_CI : ',accuracy_CI)
        print('precision_CI : ',precision_CI)
        print('recall_CI: ',recall_CI)
        print('specificity_CI: ',specificity_CI)
        print('NPV_CI: ',NPV_CI)
        print('kappa: ',kap)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        specificitys.append(specificity)
        recall_CIs.append(recall_CI)
        specificity_CIs.append(specificity_CI)
#specificitys[0]
    if return_single:
        return precisions[0], recalls[0], f1_scores[0], specificitys[0] ,recall_CIs[0],specificity_CIs[0]
    else:
        return precisions, recalls, f1_scores, specificitys,recall_CIs,specificity_CIs


def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res
