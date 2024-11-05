import numpy as np


def gram_schmidt(X):
    # Aplicamos el proceso de Gram-Schmidt a la matriz X para obtener Q y R
    n, m = X.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = X[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError(
                "Las columnas de la matriz X son linealmente dependientes o contienen una columna de ceros.")
        Q[:, j] = v / R[j, j]

    return Q, R


def linear_regression_gs(X, y):
    # Asegurarse de que X y y tienen dimensiones compatibles
    if X.shape[0] != y.shape[0]:
        raise ValueError("El número de filas de X debe ser igual al tamaño de y.")

    # Añadir una columna de unos a X para el término independiente
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Obtener las matrices Q y R mediante Gram-Schmidt
    Q, R = gram_schmidt(X)

    # Calcular Q^T * y
    Qt_y = np.dot(Q.T, y)

    # Resolver R * beta = Q^T * y mediante sustitución hacia atrás
    beta = np.zeros(R.shape[1])
    for i in range(R.shape[1] - 1, -1, -1):
        if R[i, i] == 0:
            raise ValueError("La matriz R es singular, no se puede resolver el sistema.")
        beta[i] = (Qt_y[i] - np.dot(R[i, i + 1:], beta[i + 1:])) / R[i, i]

    return beta



