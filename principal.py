from Cholesky import *
if __name__ == "__main__":
    X = [
        [4, 1, 3, 5, 7, 2],
        [1, 6, 4, 8, 3, 5],
        [3, 4, 9, 6, 2, 7],
        [5, 8, 6, 7, 4, 3],
        [7, 3, 2, 4, 8, 6],
        [2, 5, 7, 3, 6, 9]
    ]
    y = [1, 2,1,2,21,1]

    try:
        beta = regresionLinealCholesky(X, y)
        print("Coeficientes beta obtenidos:", beta)
    except ValueError as e:
        print(e)