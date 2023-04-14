import sys 
import numpy as np
from config import path
sys.path.append(path)
from dezero import Variable

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward() #import dezero.funtions in __init__.py
print(x1.grad)  