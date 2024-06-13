from SCORE import SCORE
import matplotlib.pyplot as plt
import numpy as np

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    """
    x = np.transpose(x)
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

bounds = np.arange(-5, 10.01, 0.05)
parameters = [{'name': f'x{i}', 'scale': 'linear', 'type': 'discrete', 'domain': bounds} for i in range(10)]

n_init = 20
init_combs = None
af = 'EI'
xi = 0.1

model = SCORE(parameters=parameters, f=ackley, n_init=n_init, init_combs=init_combs, af='EI', xi=xi)

nb_it = 500
n_cbs = 1

min_t, bo = model.fit(nb_it=nb_it, n_cbs=n_cbs, verbose=False)

print(min_t)
model.plot()
print(model.return_min())
plt.show()

