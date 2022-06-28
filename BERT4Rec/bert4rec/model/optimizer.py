import abc
import math

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

from bert4rec.utils import get_logger


logger = get_logger()


class _LRSchedule(abc.ABC):
    warn_t_total = False

    def __init__(self, warmup=0.002, t_total=-1, **kwargs):
        super(_LRSchedule, self).__init__(**kwargs)
        if t_total < 0:
            logger.warning(f"t_total value of {t_total} results in schedule not being applied")
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(f"Invalid warmup: {warmup} - should be in [0.0, 1.0) or -1")
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            logger.warning(
                f"Training beyond specified 't_total'. Learning rate multiplier set to {ret}. "
                f"Please set 't_total' of {self.__class__.__name__} correctly."
            )
            self.warned_for_t_total_at_progress = progress
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        return 1.


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    warn_t_total = True

    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kwargs):
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kwargs)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kwargs):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(warmup=warmup,
                                                                   t_total=t_total,
                                                                   cycles=cycles,
                                                                   **kwargs)
        assert(cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)
            ret = 0.5 * (1. + math.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kwargs):
        assert(warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(warmup=warmup,
                                                                     t_total=t_total,
                                                                     cycles=cycles,
                                                                     **kwargs)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    warn_t_total = True

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None: ConstantLR,
    "none": ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}


class BERTAdam(Optimizer):
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule="warmup_linear",
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError(f"Invalid schedule parameter: {schedule}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid b1 parameter: {b1} - should be in [0.0, 1.0)")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid b2 parameter: {b2} - should be in [0.0, 1.0)")
        if not e >= 0.0:
            raise ValueError(f"Invalid epsilon value: {e} - should be >= 0.0")
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning(
                    "warmup and t_total on the optimizer are ineffective "
                    "when _LRSchedule object is provided as schedule. "
                    "Please specify custom warmup and t_total in _LRSchedule object."
                )
        defaults = dict(lr=lr, schedule=schedule,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["next_m"] = torch.zeros_like(p.data)
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group["e"])

                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state["step"] += 1

        return loss
