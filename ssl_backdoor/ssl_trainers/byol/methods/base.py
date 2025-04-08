import os
import torch.nn as nn
from model import get_model, get_head
from eval.sgd import eval_sgd
from eval.knn import eval_knn
from eval.get_data import get_data


class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model, self.out_size = get_model(cfg.arch, cfg.dataset)
        self.head = get_head(self.out_size, cfg)
        self.knn = cfg.knn
        self.num_pairs = cfg.num_samples * (cfg.num_samples - 1) // 2
        self.eval_head = cfg.eval_head
        self.emb_size = cfg.emb

        self.cfg = cfg

    def forward(self, samples):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test, ds_test_p):
        self.eval()
        if self.eval_head:
            model = lambda x: self.head(self.model(x))
            out_size = self.emb_size
        else:
            model, out_size = self.model, self.out_size
        # torch.cuda.empty_cache()
        x_train, y_train = get_data(model, ds_clf, out_size, "cuda")
        x_test, y_test = get_data(model, ds_test, out_size, "cuda")
        x_test_p, y_test_p = get_data(model, ds_test_p, out_size, "cuda")

        acc_knn = eval_knn(x_train, y_train, x_test, y_test, self.knn)
        asr_knn = eval_knn(x_train, y_train, x_test_p, y_test_p, self.knn)
        acc, y_pred, y_test, acc_p, y_pred_p, y_test_p = eval_sgd(model, ds_clf, ds_test, ds_test_p, out_size, "cuda")

        
        del x_train, y_train, x_test, y_test, x_test_p, y_test_p
        self.train()
        return acc_knn, acc, asr_knn, acc_p

    def step(self, progress):
        pass
