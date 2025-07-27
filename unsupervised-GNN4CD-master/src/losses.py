import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor


def from_scores_to_labels_multiclass_batch(pred):
    labels_pred = np.argmax(pred, axis = 2).astype(int)
    return labels_pred

def compute_accuracy_multiclass_batch(labels_pred, labels):
    overlap = (labels_pred == labels).astype(int)
    acc = np.mean(labels_pred == labels)
    return acc

def compute_loss_multiclass(pred_llh, labels, n_classes):
    loss = 0
    permutations = permuteposs(n_classes)
    batch_size = pred_llh.data.cpu().shape[0]
    for i in range(batch_size):
        pred_llh_single = pred_llh[i, :, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            labels_under_perm = torch.from_numpy(permutations[j, labels_single.data.cpu().numpy().astype(int)])
            loss_under_perm = criterion(pred_llh_single, labels_under_perm.type(dtype_l))

            if (j == 0):
                loss_single = loss_under_perm
            else:
                loss_single = torch.min(loss_single, loss_under_perm)
        loss += loss_single
    return loss

def modularity_loss(logits, A):
    Y = F.softmax(logits, dim=1)  #The sum of every row is equal to 1

    degrees = A.sum(dim=1)
    m = degrees.sum() / 2
    YYT = torch.matmul(Y, Y.T)
    expected = torch.outer(degrees, degrees) / (2 * m)
    B = A - expected
    Q = (YYT * B).sum() / (2 * m)
    return -Q

def compute_modularity_loss_multiclass(pred_llh, adj_batch):
    """
    pred_llh: [batch_size, n_nodes, n_clusters] — GNN logits
    adj_batch: [batch_size, n_nodes, n_nodes] — adjacency matrix per graph
    return: modularity score (the higher the better)
    """
    if not isinstance(adj_batch, torch.Tensor):
        adj_batch = torch.from_numpy(adj_batch.toarray() if hasattr(adj_batch, "toarray") else adj_batch).float()

    loss = 0.0
    batch_size = pred_llh.shape[0]

    for i in range(batch_size):
        logits = pred_llh[i]               # [n_nodes, n_clusters]
        A = adj_batch[i]                  # [n_nodes, n_nodes]

        # Convert logits to soft cluster assignment
        Y = F.softmax(logits, dim=1)      # [n_nodes, n_clusters]

        # Degree & total edges
        degrees = A.sum(dim=1)            # [n]
        m = degrees.sum() / 2             # scalar

        # Soft cluster similarity
        YYT = torch.matmul(Y, Y.T)        # [n_nodes, n_nodes]

        # Expected connections under null model
        expected = torch.outer(degrees, degrees) / (2 * m)

        B = A - expected
        Q = (YYT * B).sum() / (2 * m)     # modularity

        loss += Q                         # We want to maximize modularity

    return  -loss / batch_size              # Average over batch

# def compute_accuracy_multiclass(pred_llh, labels, n_classes):
#     pred_llh = pred_llh.data.cpu().numpy()
#     labels = labels.data.cpu().numpy()
#     batch_size = pred_llh.shape[0]
#     pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
#     acc = 0
#     permutations = permuteposs(n_classes)
#     for i in range(batch_size):
#         pred_labels_single = pred_labels[i, :]
#         labels_single = labels[i, :]
#         for j in range(permutations.shape[0]):
#             permutation = permutations[j, :]
#             labels_under_perm = permutations[j, labels_single.astype(int)]
#
#             acc_under_perm = compute_accuracy_multiclass_batch(pred_labels_single, labels_under_perm)
#             if (j == 0):
#                 acc_single = acc_under_perm
#             else:
#                 acc_single = np.max([acc_single, acc_under_perm])
#
#         acc += acc_single
#     acc = acc / labels.shape[0]
#     acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)
#     return acc
def compute_accuracy_multiclass(pred_llh, labels, n_classes):
    pred_llh = pred_llh.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    batch_size = pred_llh.shape[0]
    pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
    permutations = permuteposs(n_classes)

    best_matched_preds = np.zeros_like(labels)
    acc_total = 0

    for i in range(batch_size):
        pred_labels_single = pred_labels[i, :]
        labels_single = labels[i, :]

        best_acc = -1
        best_perm = None

        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            # Apply permutation to prediction instead of label
            pred_perm = permutation[pred_labels_single.astype(int)]
            acc_under_perm = compute_accuracy_multiclass_batch(pred_perm, labels_single)

            if acc_under_perm > best_acc:
                best_acc = acc_under_perm
                best_perm = permutation

        # 最优 permutation 作用于 prediction，确保其标签顺序与 labels 一致
        best_matched_preds[i, :] = best_perm[pred_labels_single.astype(int)]
        acc_total += best_acc

    acc = acc_total / batch_size
    # acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)  # Normalized Accuracy

    return acc, best_matched_preds

def permuteposs(n_classes):
    permutor = Permutor(n_classes)
    permutations = permutor.return_permutations()
    return permutations


class Permutor:
    def __init__(self, n_classes):
        self.row = 0
        self.n_classes = n_classes
        self.collection = np.zeros([math.factorial(n_classes), n_classes])

    def permute(self, arr, l, r): 
        if l==r: 
            self.collection[self.row, :] = arr
            self.row += 1
        else: 
            for i in range(l,r+1): 
                arr[l], arr[i] = arr[i], arr[l] 
                self.permute(arr, l+1, r) 
                arr[l], arr[i] = arr[i], arr[l]

    def return_permutations(self):
        self.permute(np.arange(self.n_classes), 0, self.n_classes-1)
        return self.collection
                
