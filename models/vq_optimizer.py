import torch
from models.vq_ops import code_books, get_code_book, vq


class VQSGD(torch.optim.SGD):
    def __init__(self, myargs, *args, **kwargs):
        super(VQSGD, self).__init__(*args, **kwargs)
        print('VQSGD')
        self.args = myargs
        self.dim = myargs.dim
        self.ks = myargs.ks
        self.rate = 1
        # self.code_book is of dim * ks
        self.code_book = get_code_book(myargs, self.dim, self.ks)

    def __setstate__(self, state):
        super(VQSGD, self).__setstate__(state)

    def gd(self, p, lr, d_p):
        p.data.add_(-lr, d_p)

    def vq_gd(self, p, lr, d_p):
        if p.data.reshape(-1).size(0) < 1024:
            self.gd(p, lr, d_p)
        else:
            l = 1 / self.rate * lr
            u = self.rate * lr
            # codebook should be of size dim * ks
            x = p.data.reshape(-1, self.dim, 1)
            grad = d_p.reshape(-1, self.dim, 1)

            M = x.size(0)
            W = x.expand(-1, -1, self.ks)
            F = grad.expand(-1, -1, self.ks)
            C = self.code_book.t().expand(M, self.dim, self.ks)
            Tu = C - W + F.mul(u)
            Tl = C - W + F.mul(l)
            # result should be of shape M * dim * ks
            r0 = torch.min(Tu.pow(2), Tl.pow(2))
            r1 = (torch.sign(Tu) + torch.sign(Tl)).pow(2)
            result = 1/4 * r0.mul(r1)
            codes = result.sum(dim=1).argmin(dim=1)

            # p.data.add_(-lr, d_p)
            # x1 = p.data.view(-1, self.dim)
            # codes1 = vq(x1, self.code_book)
            # print(torch.sum(codes == codes1), codes1.size(0))
            torch.index_select(
                self.code_book, 0, codes, out=x[:, :, :])
            p.data = x.view(p.data.shape)

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
                # the gradient of p
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

                # torch.add(input, alpha=1, other, out=None) -> out = input + alpha * other
                # p.data.add_(-group['lr'], d_p)
                # self.gd(p, group['lr'], d_p)
                self.vq_gd(p, group['lr'], d_p)

        return loss
