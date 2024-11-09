"""import numpy as np

def add_intercept_column(X):

    #Agrega una columna de unos a la matriz X para incluir el término independiente en el modelo.

    n = X.shape[0]
    intercept = np.ones((n, 1))  # Columna de unos
    return np.hstack((intercept, X))  # Concatenamos la columna de unos a X

def check_conditions(X, y):
    n, p = X.shape
    if n < p:
        raise ValueError("El número de observaciones debe ser mayor o igual al número de variables predictoras (n >= p).")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("Las columnas de X deben ser linealmente independientes (rango completo).")
    if y.shape[0] != n:
        raise ValueError("El vector y debe tener el mismo número de filas que X.")
    print("Todas las condiciones se cumplen.")
    return True

def gram_schmidt(X):
    n, p = X.shape
    Q = np.zeros((n, p))
    for j in range(p):
        q = X[:, j]
        for i in range(j):
            q = q - np.dot(Q[:, i], X[:, j]) * Q[:, i]
        Q[:, j] = q / np.linalg.norm(q)
    return Q

def regression_coefficients(X, y):
    check_conditions(X, y)
    Q = gram_schmidt(X)
    beta = np.dot(Q.T, y) / np.sum(Q**2, axis=0)
    return beta

def calculate_errors(X, y, beta):
    y_pred = X @ beta
    mse = np.mean((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, r2


def print_regression_equation(beta, include_intercept=True):

    #Imprime la ecuación de regresión en formato legible.

    equation = "y = "
    if include_intercept:
        # El primer coeficiente es el intercepto
        equation += f"{beta[0]:.4f} "
        start = 1
    else:
        start = 0

    for i in range(start, len(beta)):
        sign = "+" if beta[i] >= 0 else "-"
        equation += f"{sign} {abs(beta[i]):.4f} * x{i}"
        if i < len(beta) - 1:
            equation += " "

    print("Ecuación de regresión:", equation)"""
import numpy as np

def add_intercept_column(X):
    n = X.shape[0]
    intercept = np.ones((n, 1), dtype=np.float64)  # Aseguramos precisión con np.float64
    return np.hstack((intercept, X.astype(np.float64)))  # Convertimos X a np.float64

def check_conditions(X, y):
    n, p = X.shape
    if n < p:
        raise ValueError("El número de observaciones debe ser mayor o igual al número de variables predictoras (n >= p).")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("Las columnas de X deben ser linealmente independientes (rango completo).")
    if y.shape[0] != n:
        raise ValueError("El vector y debe tener el mismo número de filas que X.")
    print("Todas las condiciones se cumplen.")
    return True

def gram_schmidt(X):
    n, p = X.shape
    Q = np.zeros((n, p), dtype=np.float64)  # Aseguramos precisión con np.float64
    for j in range(p):
        q = X[:, j].astype(np.float64)
        for i in range(j):
            q = q - np.dot(Q[:, i], X[:, j]) * Q[:, i]
        Q[:, j] = q / np.linalg.norm(q)
    return Q

def regression_coefficients(X, y):
    check_conditions(X, y)
    Q = gram_schmidt(X)
    beta = np.dot(Q.T, y.astype(np.float64)) / np.sum(Q**2, axis=0)
    return beta

def calculate_errors(X, y, beta):
    y_pred = X @ beta
    mse = np.mean((y.astype(np.float64) - y_pred) ** 2)
    ss_total = np.sum((y.astype(np.float64) - np.mean(y)) ** 2)
    ss_residual = np.sum((y.astype(np.float64) - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, r2

def print_regression_equation(beta, include_intercept=True):
    equation = "y = "
    if include_intercept:
        equation += f"{beta[0]:.4f} "
        start = 1
    else:
        start = 0

    for i in range(start, len(beta)):
        sign = "+" if beta[i] >= 0 else "-"
        equation += f"{sign} {abs(beta[i]):.4f} * x{i}"
        if i < len(beta) - 1:
            equation += " "

    print("Ecuación de regresión:", equation)

