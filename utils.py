import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def mask_generator(shape, p_m):
    """Generate mask vector.

    Args:
    - p_m: corruption probability
    - x: feature matrix

    Returns:
    - mask: binary mask matrix
    """
    mask = torch.bernoulli(torch.ones(shape) * p_m)
    return mask.to(torch.float)


def pretext_generator(m, x):
    """Generate corrupted samples.

    Args:
    m: mask matrix
    x: feature matrix

    Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
    """

    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = torch.zeros([no, dim]).to(x.device)
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)
    return m_new.to(torch.float), x_tilde.to(torch.float)


def perf_metric(metric, y_test, y_test_hat):
    """Evaluate performance.

    Args:
      - metric: acc or auc
      - y_test: ground truth label
      - y_test_hat: predicted values

    Returns:
      - performance: Accuracy or AUROC performance
    """
    # Accuracy metric
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis=1),
                                np.argmax(y_test_hat, axis=1))
    # AUROC metric
    elif metric == 'auc':
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])

    return result


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
