import sys
#from dezero.config import path
path = 'c:/Users/Nayoung/OneDrive/Study/AI/BOOK/DL/DL_from_scratch_vol3/code/dezero/dezero'
sys.path.append(path)

import numpy as np
from dezero.core import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data