from GramSchmidt import *
from TriangulaciónHouseholder import *



if __name__ == '__main__':
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    y = np.array([1, 2, 3, 4])
    print(X)
    print(y)
    try:
        # Estimar el vector de coeficientes beta
        beta = linear_regression_gs(X, y)
        print("Coeficientes estimados:", beta)
    except ValueError as e:
        print("Error:", e)