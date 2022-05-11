from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class DecreasingCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, factor, epochs, min_lr=0, eps=1e-8, **kwargs):
        super(DecreasingCosineAnnealingWarmRestarts, self).__init__(**kwargs)
        self.factor = factor
        self.epochs = epochs
        self.eps = eps
        self.min_lrs = [min_lr] * len(self.optimizer.param_groups)
        self.current_epoch = 0

    def step(self, epoch=None):
        super().step(epoch=epoch)
        if self.T_cur == 0:
            self.current_epoch += 1
        if self.current_epoch in self.epochs:
            self._reduce_lr()

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
