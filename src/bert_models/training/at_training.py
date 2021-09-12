
import logging
import re

import torch


logger = logging.getLogger(__name__)


class FGM(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""
    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],
                 epsilon=1.0):
        self.model = model

        # emb_names 这个参数要换成你模型中embedding的参数名
        # 可以是多组参数
        self.emb_names = emb_names

        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                # 把真实参数保存起来
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]



class PGD(object):
    """Reference: https://arxiv.org/pdf/1706.06083.pdf"""
    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],
                 epsilon=1.0,
                 alpha=0.3):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]