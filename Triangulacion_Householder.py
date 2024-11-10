import numpy as np

def check_conditions_householder(X, Y):
    n_samples, n_features = X.shape
    if Y.shape[0] != n_samples:
        raise ValueError("El vector Y debe tener el mismo número de filas que X.")
    # Verificar el rango de X (opcional)
    rank_X = np.linalg.matrix_rank(X)
    if rank_X < n_features:
        raise ValueError("La matriz X no tiene rango completo. Puede haber multicolinealidad.")
    print("Las matrices X e Y cumplen con las condiciones necesarias.")

def householder_transformation(A):
    """
    Aplica la Triangulación de Householder a la matriz A y devuelve Q y R.
    """
    (rows, cols) = A.shape
    Q = np.identity(rows)
    R = A.copy()

    for k in range(cols):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        s = -np.sign(x[0]) if x[0] != 0 else -1
        u1 = x[0] - s * norm_x
        v = x.copy()
        v[0] = u1
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        v = v / norm_v
        H = np.identity(rows)
        H[k:, k:] -= 2 * np.outer(v, v)
        R = H @ R
        Q = Q @ H.T
    return Q, R


def regression_coefficients_householder(X, Y):
    """
    Calcula los coeficientes de regresión usando la Triangulación de Householder.
    """
    # Verificar las condiciones de X e Y
    check_conditions_householder(X, Y)

    # Agregar una columna de unos para el intercepto
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Aplicar la Triangulación de Householder
    Q, R = householder_transformation(X)

    # Calcular Q^T * Y
    Q_T_Y = Q.T @ Y

    # Resolver el sistema triangular superior R * beta = Q^T * Y
    n = R.shape[1]
    beta = np.linalg.solve(R[:n, :], Q_T_Y[:n])
    return beta


def calculate_errors_householder(X, Y, beta):
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


def print_regression_equation(beta, include_intercept=True):
    """Imprime la ecuación de regresión en formato legible."""
    equation = "Y = "
    if include_intercept:
        equation += f"{beta[0]:.4f} "
        for i, b in enumerate(beta[1:], start=1):
            equation += f"+ ({b:.4f}) * X{i} "
    else:
        for i, b in enumerate(beta, start=1):
            equation += f"({b:.4f}) * X{i} "
    print("Ecuación de regresión:", equation)
