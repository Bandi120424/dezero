import sys
from dezero.config import path
sys.path.append(path)

import numpy as np
from dezero.core_simple import as_variable
from dezero.core_simple import Variable
from dezero.utils import _dot_var, _dot_func, plot_dot_graph

x = Variable(np.random.randn(2, 3))
x.name = 'x'
print("=============================")
print("_dot_var(x)", _dot_var(x))
print("_dot_var(x, verbose=True)", _dot_var(x, verbose=True))

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0+x1
txt = _dot_func(y.creator)
print("=============================")
print("_dot_func result", txt)


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()

x.name = 'x'
y.name = 'y'
z.name = 'z'
print("=============================")
plot_dot_graph(z, verbose=False, to_file='goldstein.png')
