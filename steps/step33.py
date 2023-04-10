import sys 
from config import path
sys.path.append(path)
    
import numpy as np
from dezero.core import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)

iters = 10

for i in range(iters):  
    print(i, x)

    y = f(x)
    x.cleargrad() #미분값 재설정
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad() #미분값 재설정
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
