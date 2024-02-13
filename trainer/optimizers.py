import torch
import torch.optim as optim


def make_optimizer(model: torch.nn.Module, learning_rate: float, config: dict):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, **config)
    return optimizer
