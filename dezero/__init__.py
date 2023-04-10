import sys
from config import path
#sys.path.append(path)

is_simple_core = False

if is_simple_core:
    #print("okay")
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
    
else:
    from dezero.core import Variable
    #from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    #from dezero.core import test_mode
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    #from dezero.core import Config
    '''
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.datasets import Dataset
    from dezero.dataloaders import DataLoader
    from dezero.dataloaders import SeqDataLoader

    import dezero.datasets
    import dezero.dataloaders
    import dezero.optimizers
    import dezero.functions
    import dezero.functions_conv
    import dezero.layers
    import dezero.utils
    import dezero.cuda
    import dezero.transforms
    '''

'''
Variable의 연산자들을 설정 
dezero 패키지에서는 반드시 연산자 오버로드가 이루어진 상태에서 Variable을 사용
'''
setup_variable()
__version__ = '0.0.13'
