import numpy as np

def check_conditions_svd(X, y):
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("El vector y debe tener el mismo número de filas que X.")
    print("Condiciones de formato se cumplen.")

def regression_coefficients_svd2(X, Y):
    check_conditions_svd(X, Y)
    # Agregar la columna de unos para el intercepto
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Descomposición SVD de X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Invertir los valores singulares
    S_inv = np.diag(1 / s)

    # Calcular los coeficientes beta
    beta = Vt.T @ S_inv @ U.T @ Y
    return beta

def calculate_errors2(X, Y, beta):
    # Asegurarse de que X tenga la columna de unos (intercepto)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Predicciones de Y
    Y_pred = X @ beta

    # Error cuadrático medio
    mse = np.mean((Y - Y_pred) ** 2)

    # Coeficiente de determinación R^2
    ss_total = np.sum((Y - np.mean(Y)) ** 2)
    ss_residual = np.sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2

def print_regression_equation2(beta, include_intercept=True):
    equation = "Y = "
    if include_intercept:
        equation += f"{beta[0]:.4f} "
        for i, b in enumerate(beta[1:], start=1):
            equation += f"+ ({b:.4f}) * X{i} "
    else:
        for i, b in enumerate(beta, start=1):
            equation += f"({b:.4f}) * X{i} "

    print("Ecuación de regresión:", equation)