import numpy as np

def predict(matrix: np.ndarray, p1, p2, p3, error=False):
    row, col = matrix.shape
    ret = np.copy(matrix)
    
    def getPredictors(ri, ci):
        if ri == 0 and ci == 0:
            return matrix[ri, ci], 0, 0
        elif ri == 0:
            return matrix[ri, ci-1], 0, 0
        elif ci == 0:
            return 0, matrix[ri-1, ci], 0
        return matrix[ri, ci-1], matrix[ri-1, ci], matrix[ri-1, ci-1]
    
    for j in range(1, col):
        if error:
            ret[0, j] -= matrix[0, j-1]
        else:
            ret[0, j] = matrix[0, j-1]
            
    for i in range(1, row):
        if error:
            ret[i, 0] -= matrix[i-1, 0]
        else:
            ret[i, 0] = matrix[i-1, 0]

    for i in range(1, row):
        for j in range(1, col):
            a, b, c = getPredictors(i, j)
            if error:
                ret[i, j] -= p3(a, b, c)
            else:
                ret[i, j] = p3(a, b, c)
                
    return ret
def entropy(matrix: np.ndarray):
    unique, counts = np.unique(matrix, return_counts=True)
    ret = np.sum(counts / matrix.size * np.log2(counts / matrix.size))
    print("".join(f"{u:>6}" for u in unique))
    print("".join(f"{u:>6}" for u in counts))
    return ret

if __name__ == '__main__':
    A = lambda a, b=None, c=None : a
    B = lambda a, b, c=None : b
    MAX = lambda a, b, c : max(a, b, c, a+b-c)
    
    matrix = np.array([
        [15, 15, 12, 11],
        [13, 16, 18, 14],
        [14, 14, 15, 16],
        [13, 12, 17, 17],
    ])
    predicted = predict(matrix, A, B, MAX)
    predict_error = predict(matrix, A, B, MAX, error=True)
    print(f"""
original matrix: {entropy(matrix)}
predicted matrix: {entropy(predict_error)} 
          """)
    print(predicted)
    print(predict_error)