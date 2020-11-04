
def dot(v, w):
    return sum([*map( lambda x: x[0] * x[1], zip(v, w))])
