import numpy as np
import os
import subprocess
from dezero.core_simple import as_variable
from dezero.core_simple import Variable


# =============================================================================
# Visualize for computational graph
# =============================================================================

def _dot_var(v, verbose=False):
    """graph의 각 원소를 생성하는 DOT 언어 구문을 작성
    - Variable instance로 받은 값을 DOT 언어로 작성된 문자열로 바꿔 반환
    - get_dot_graph에서만 사용하는 함수

    Args:
        v (_class_Variable): graph의 원소가 될 instance 값 
        verbose (bool, optional): 출력결과에 v의 shape와 type을 포함할지의 여부. Defaults to False.

    Returns:
        str : DOT 구문
    """
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)

def _dot_func(f):
    """DeZero 함수 => DOT 언어 
       함수의 input들과 output의 관계를 화살표로 매핑
    Args:
        f (_class_Function): DOT 언어로 변환할 함수 
    Returns:
        str : DOT 언어
    """    
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret

def get_dot_graph(output, verbose=True):
    """computational graph를 생성하는 graphviz DOT text 생성
    output에서부터 역방향으로 function과 variable을 탐색해가며 graph를 그림 
    (Build a graph of functions and variables backward-reachable from the
    output)
    Args:
        output (dezero.Variable): computational graph를 생성할 Output variable 
                                  (computational graph의 output)
        verbose (bool): shapes과 dtypes 정보 포함 여부
    Returns:
        str: nodes와 edges로 구성된 graphviz DOT text 
        (nodes & edges of computational graph built by backward-reachable from the output)
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    """ouput을 생성하는 computational graph를 생성 후, 이미지 파일로 저장 

    Args:
        output (dezero.Variable): computational graph를 생성할 Output variable 
                                  (computational graph의 output)
        verbose (bool): shapes과 dtypes 정보 포함 여부 (default = True)
        to_file (str, optional): computational graph 이미지 파일명 (default = 'graph.png')

    Returns:
        img_file : computational graph 이미지 파일
    """    
    dot_graph = get_dot_graph(output, verbose)

    #DOT 데이터 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero') #~/.dezero 디렉토리 없으면 새로 생성
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    #DOT 명령 호출
    extension = os.path.splitext(to_file)[1][1:]  # Extension(e.g. png, pdf)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # Return the image as a Jupyter Image object, to be displayed in-line.
    # try:
    #     from IPython import display
    #     return display.Image(filename=to_file)
    # except:
    #     pass

def sum_to(x, shape):
    """주어진 shape대로 두 elements를 더함 (Sum elements along axes to output an array of a given shape)
    Args:
        x (ndarray): Input array.
        shape:
    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """dezero.functions.sum의 역전파 수행을 위해 그레디언트를 reshape
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy