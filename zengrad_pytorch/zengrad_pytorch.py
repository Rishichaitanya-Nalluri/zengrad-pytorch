import torch
from torch.optim import Optimizer


class ZenGrad(Optimizer):
    """ZenGrad with Nesterov momentum only:
        A_t   = A_{t-1} + g_t^2
        v_t   = mu * v_{t-1} + g_t
        lr_t  = lr / (log(A_t + 1) + eps)
        p    <- p * (1 - lr * wd) - lr_t * (g_t + mu * v_t)
    """

    def __init__(self, params, lr=0.1, initial_accumulator_value=1.718281828,
                 weight_decay=5e-4, epsilon=1e-8, momentum=0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid accumulator value: {initial_accumulator_value}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if momentum <= 0.0 or momentum >= 1.0:
            raise ValueError(
                f"Invalid momentum value: {momentum} (Nesterov requires 0 < momentum < 1)"
            )

        defaults = dict(
            lr=lr,
            initial_accumulator_value=initial_accumulator_value,
            weight_decay=weight_decay,
            epsilon=epsilon,
            momentum=momentum,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            acc_init = group['initial_accumulator_value']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ZenGrad does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if 'accumulator' not in state:
                    state['accumulator'] = torch.full_like(p, acc_init)
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                accumulator = state['accumulator']
                buf = state['momentum_buffer']

                # Accumulate squared gradients
                accumulator.add_(grad.pow(2))

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Effective learning rate
                effective_lr = lr / (torch.log1p(accumulator) + epsilon)

                # Nesterov momentum: v_t = mu * v_{t-1} + g_t ; step uses g_t + mu * v_t
                buf.mul_(momentum).add_(grad)
                p.addcmul_(grad + momentum * buf, -effective_lr)

        return loss