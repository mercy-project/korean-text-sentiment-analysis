from scipy.spatial.distance import cdist

def euclidean(xa, xb):
    dist = cdist(xa, xb, 'euclidean')
    return dist

def cosine(xa, xb):
    dist = cdist(xa, xb, 'cosine')
    return dist