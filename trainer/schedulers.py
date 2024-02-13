import torch

class WarmupScheduler:
    def __init__(self, scheduler, lr_min, warmup):
        self.scheduler = scheduler
        self.warmup = warmup
        self.lr_min = lr_min
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(scheduler.optimizer,
                                                                  lambda epoch: min(epoch / warmup, 1))
        self.current_epoch = 0

    def step(self, metrics=None):
        if self.current_epoch < self.warmup:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step(metrics)
        self.current_epoch += 1

    def get_lr(self):
        if self.current_epoch < self.warmup:
            scheduler = self.warmup_scheduler
        else:
            scheduler = self.scheduler
        return [params['lr'] for params in scheduler.optimizer.param_groups]

    def state_dict(self):
        state_dict = self.scheduler.state_dict()
        state_dict['last_epoch_with_warmup'] = self.current_epoch
        return state_dict

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['last_epoch_with_warmup']
        state_dict.pop('last_epoch_with_warmup')
        self.scheduler.load_state_dict(state_dict)


def make_lr_scheduler(kind, optimizer, kwargs, warmup, config):
    lr = config['learning_rate']
    lr_min = config['learning_rate_min']
    epochs = config['num_epochs']

    if warmup > 0:
        epochs = epochs - warmup - 1

    if kind == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1)
    elif kind == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    elif kind == 'linear':
        lr_ratio = lr_min / lr
        func = lambda epoch: epoch / epochs * (lr_ratio - 1) + 1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    elif kind == 'plateau':
        kwargs_default = dict(mode='max', factor=0.5, patience=config['patience'] // 2)
        kwargs_default.update(kwargs)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs_default)
    else:
        raise ValueError(f"Unknown kind of lr scheduler: {kind}")

    if warmup > 0:
        lr_scheduler = WarmupScheduler(lr_scheduler, lr_min, warmup)

    return lr_scheduler
