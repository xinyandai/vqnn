import torch
import math
from models.vq_ops import get_code_book


class VQSGD(torch.optim.SGD):
    def __init__(self, myargs, *args, **kwargs):
        super(VQSGD, self).__init__(*args, **kwargs)
        self.args = myargs
        self.dim = myargs.dim
        self.ks = myargs.ks
        self.rate = myargs.rate
        print('VQSGD, rate = {}'.format(self.rate))
        self.code_books = get_code_book(self.args, self.dim, self.ks)

    def vq_gd(self, p, lr, d_p):
        code_books = self.code_books
        l = 1 / self.rate * lr
        u = self.rate * lr
        x = p.data.reshape(-1, self.dim, 1)
        grad = d_p.reshape(-1, self.dim, 1)
        M = x.size(0)
        W = x.expand(-1, -1, self.ks)
        F = grad.expand(-1, -1, self.ks)
        C = code_books.t().expand(M, self.dim, self.ks)
        Tu = C - W + F.mul(u)
        Tl = C - W + F.mul(l)

        r0 = torch.min(Tu.pow(2), Tl.pow(2))
        r1 = (torch.sign(Tu) + torch.sign(Tl)).pow(2)
        result = 1/4 * r0.mul(r1)
        codes = result.sum(dim=1).argmin(dim=1)

        p1 = torch.index_select(
            code_books, 0, codes).reshape(p.data.shape)
        return p1

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                if hasattr(p, 'org'):
                    p.org.copy_(p.data - group['lr'] * d_p)
                if 'name' in group and group['name'] == 'others':
                    p.data.add_(-group['lr'], d_p)
                else:
                    if group['name'] == 'conv2d':
                        tensor = p.data.clone().detach().permute(0, 2, 3, 1).contiguous()
                        p.data = self.vq_gd(
                            tensor, group['lr'], d_p).permute(0, 3, 1, 2)
                    else:
                        p.data = self.vq_gd(p, group['lr'], d_p)

        return loss


class VQAdam(torch.optim.Adam):
    def __init__(self, myargs, *args, **kwargs):
        super(VQAdam, self).__init__(*args, **kwargs)

        self.args = myargs
        self.dim = myargs.dim
        self.ks = myargs.ks
        self.rate = myargs.rate
        print('VQAdam, rate = {}'.format(self.rate))
        self.code_books = get_code_book(self.args, self.dim, self.ks)

    def vq_gd(self, p, lr, d_p):
        code_books = self.code_books
        l = 1 / self.rate * lr
        u = self.rate * lr
        x = p.data.reshape(-1, self.dim, 1)
        grad = d_p.reshape(-1, self.dim, 1)
        M = x.size(0)
        W = x.expand(-1, -1, self.ks)
        F = grad.expand(-1, -1, self.ks)
        C = code_books.t().expand(M, self.dim, self.ks)
        Tu = C - W + F.mul(u)
        Tl = C - W + F.mul(l)

        r0 = torch.min(Tu.pow(2), Tl.pow(2))
        r1 = (torch.sign(Tu) + torch.sign(Tl)).pow(2)
        result = 1/4 * r0.mul(r1)
        codes = result.sum(dim=1).argmin(dim=1)

        p1 = torch.index_select(code_books, 0, codes).reshape(p.data.shape)
        return p1

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                if hasattr(p, 'org'):
                    p.org.copy_(p.data - step_size * (exp_avg / denom))
                if 'name' in group and group['name'] == 'others':
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    if group['name'] == 'conv2d':
                        tensor = p.data.clone().detach().permute(0, 2, 3, 1).contiguous()
                        p.data = self.vq_gd(
                            tensor, step_size, exp_avg / denom).permute(0, 3, 1, 2)
                    else:
                        p.data = self.vq_gd(
                            p, step_size, exp_avg / denom)

        return loss
