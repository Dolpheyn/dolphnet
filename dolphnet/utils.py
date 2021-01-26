def dot(v, w):
    return sum([*map( lambda x: x[0] * x[1], zip(v, w))])

def sigmoid(_in):
    from math import exp
    return 1.0 / (1.0 + exp(-_in))
