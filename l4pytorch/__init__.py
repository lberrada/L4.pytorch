import math
import torch


def sum_inner_products(iterable1, iterable2):
    assert len(iterable1) == len(iterable2)
    return sum(torch.sum(w1 * w2) for w1, w2 in zip(iterable1, iterable2))


class ExponentialMovingAverage(object):
    def __init__(self, params, decay):
        self.decay = float(decay)
        self.params = list(params)
        self.buffer = {}

    def apply(self, var_dict):
        for p, v in var_dict.items():
            if p not in self.buffer:
                self.buffer[p] = torch.zeros_like(p, requires_grad=False)
            self.buffer[p].mul_(self.decay)
            self.buffer[p].add_(1. - self.decay, v)

    def average(self, p):
        return self.buffer[p].clone()


def time_factor(time_step, global_step):
    """ Routine used for bias correction in exponential moving averages, as in (Kingma, Ba, 2015) """
    global_step = 1 + global_step
    decay = 1.0 - 1.0 / time_step
    return 1.0 - math.exp(global_step * math.log(decay))


class AdamTransform(object):
    """
    Class implementing Adam (Kingma, Ba 2015) transform of the gradient.
    """
    def __init__(self, params, time_scale_grad=10.0, time_scale_var=1000.0, epsilon=1e-4):
        self.time_scale_grad = time_scale_grad
        self.time_scale_var = time_scale_var
        self.epsilon = epsilon
        self.params = list(params)

        self.EMAgrad = ExponentialMovingAverage(self.params, decay=1.0 - 1.0 / self.time_scale_grad)
        self.EMAvar = ExponentialMovingAverage(self.params, decay=1.0 - 1.0 / self.time_scale_var)

    def __call__(self, global_step):
        grad_dict = dict((p, p.grad) for p in self.params)
        self.EMAgrad.apply(grad_dict)
        squared_grad_dict = dict((p, p.grad ** 2) for p in self.params)
        self.EMAvar.apply(squared_grad_dict)

        correction_term_1 = time_factor(self.time_scale_grad, global_step)
        avg_grads = [self.EMAgrad.average(p) / correction_term_1 for p in self.params]

        correction_term_2 = time_factor(self.time_scale_var, global_step)
        avg_squared_grads = [self.EMAvar.average(p) / correction_term_2 for p in self.params]

        return [(grad / (torch.sqrt(squared_grad) + self.epsilon)) for grad, squared_grad in zip(avg_grads, avg_squared_grads)]


class MomentumTransform(object):
    """
    Class implementing momentum transform of the gradient (here in the form of exponential moving average)
    """
    def __init__(self, params, time_momentum=10.0):
        self.time_momentum = time_momentum
        self.params = list(params)
        self.EMAgrad = ExponentialMovingAverage(self.params, decay=1.0 - 1.0 / self.time_momentum)

    def __call__(self, global_step):
        grad_dict = dict((p, p.grad) for p in self.params)
        self.EMAgrad.apply(grad_dict)

        correction_term = time_factor(self.time_momentum, global_step)
        new_grads = [self.EMAgrad.average(p) / correction_term for p in self.params]
        return new_grads


class L4General(object):
    """
    Class implementing the general L4 stepsize adaptation scheme as a TensorFlow optimizer. The method for applying
    gradients and minimizing a variable are implemented. Note that apply_gradients expects loss as an input parameter.
    """
    def __init__(self, params, fraction=0.15, minloss_factor=0.9, init_factor=0.75,
                 minloss_forget_time=1000.0, epsilon=1e-12,
                 gradient_estimator='momentum', gradient_params=None,
                 direction_estimator='adam', direction_params=None):
        """
        :param fraction: [alpha], fraction of 'optimal stepsize'
        :param minloss_factor: [gamma], fraction of min seen loss that is considered achievable
        :param init_factor: [gamma_0], fraction of initial loss used to initialize L_min
        :param minloss_forget_time:  [Tau], timescale for forgetting minimum seen loss
        :param epsilon: [epsilon], for numerical stability in the division
        :param gradient_estimator: [g], a gradient method to be used for gradient estimation
        :param gradient_params: dictionary of parameters to pass to gradient_estimator
        :param direction_estimator: [v], a gradient method used for update direction
        :param direction_params: dictionary of parameters to pass to direction_estimator
        """
        self.params = list(params)

        self.min_loss = None
        self.global_step = 0

        self.fraction = fraction
        self.minloss_factor = minloss_factor
        self.minloss_increase_rate = 1.0 + 1.0 / minloss_forget_time
        self.epsilon = epsilon
        self.init_factor = init_factor

        if not direction_params:
            direction_params = {}
        if not gradient_params:
            gradient_params = {}

        if direction_estimator == 'momentum':
            self.grad_direction = MomentumTransform(self.params, **direction_params)
        elif direction_estimator == 'adam':
            self.grad_direction = AdamTransform(self.params, **direction_params)
        else:
            raise RuntimeError("Unrecognized direction estimator ({})".format(direction_estimator))

        if gradient_estimator == 'momentum':
            self.deriv_estimate = MomentumTransform(self.params, **gradient_params)
        elif gradient_estimator == 'adam':
            self.deriv_estimate = AdamTransform(self.params, **gradient_params)
        else:
            raise RuntimeError("Unrecognized gradient estimator ({})".format(gradient_estimator))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data.zero_()

    def step(self, closure):
        loss = float(closure())

        if self.global_step == 0:
            self.min_loss = self.init_factor * loss
        else:
            self.min_loss = min(self.min_loss, loss)

        directions = self.grad_direction(self.global_step)
        derivatives = self.deriv_estimate(self.global_step)

        min_loss_to_use = self.minloss_factor * self.min_loss
        l_rate = self.fraction * (loss - min_loss_to_use) / (sum_inner_products(directions, derivatives) + self.epsilon)

        self.min_loss *= self.minloss_increase_rate

        for (p, grad) in zip(self.params, directions):
            p.data.add_(-float(l_rate), grad)

        self.global_step += 1


class L4Adam(L4General):
    """
    Specialization of the L4 stepsize adaptation with Adam used for gradient updates and Mom for gradient estimation.
    """
    def __init__(self, params, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
                 epsilon=1e-12, adam_params=None):
        L4General.__init__(self, params, fraction, minloss_factor, init_factor, minloss_forget_time,
                           epsilon, gradient_estimator='momentum', direction_estimator='adam',
                           direction_params=adam_params)


class L4Mom(L4General):
    """
    Specialization of the L4 stepsize adaptation with Mom used for both gradient estimation and an update direction.
    """
    def __init__(self, params, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
                 epsilon=1e-12, mom_params=None):
        L4General.__init__(self, params, fraction, minloss_factor, init_factor, minloss_forget_time,
                           epsilon, gradient_estimator='momentum', direction_estimator='momentum',
                           direction_params=mom_params)
