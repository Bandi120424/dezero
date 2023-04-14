import sys 
import numpy as np
from config import path
sys.path.append(path)
from dezero import Variable
import dezero.functions as F 

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
y.backward(retain_grad=True)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x) 
y.backward()
print(x.grad)