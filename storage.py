import torch
from torch import nn
from torch.optim import optimizer
from typing import Tuple

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"


def load_checkpoint(checkpoint_path: str, model: nn.Module, optim: optimizer.Optimizer) -> Tuple[int, int, float]:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    step = checkpoint[_STEP] + 1
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, step, best_score


def save_checkpoint(model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, step: int, best_score: float):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score
    }, "checkpoint.tar")
