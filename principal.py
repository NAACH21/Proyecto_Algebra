from Cholesky import *
if __name__ == "__main__":
    X = [
        [1, 1],
        [1, 2],
        [1, 3]
    ]
    y = [1, 2, 3]

    try:
        beta = regresionLinealCholesky(X, y)
        print("Coeficientes beta obtenidos:", beta)
    except ValueError as e:
        print(e)