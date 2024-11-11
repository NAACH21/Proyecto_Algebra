import numpy as np
#validaciones para la matriz
def verificarSimetriaPositiva(matriz):
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("Error: la matriz no es cuadrada.")

    if not np.allclose(matriz, matriz.T):
        raise ValueError("Error: la matriz no es simétrica.")

    try:
        np.linalg.cholesky(matriz)
    except np.linalg.LinAlgError:
        raise ValueError("Error: la matriz no es definida positiva.")

    return True

#la libreria numpy ya tiene integrada  la descomposicion de cholesky
#acá devuele la matris L de  X=L*LT
def descomposicionCholesky(matriz):
    return np.linalg.cholesky(matriz)
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

# resuelve un sistema de ecuaciones usando la descomposición de Cholesky
# resuelve L*y_intermedio = b, luego L^T*x = y_intermedio y obtiene x
def resolverSistemaCholesky(L, b):
    # Cambiamos el nombre de la variable para evitar colisiones
    y_intermedio = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y_intermedio)
    return x
# regresión lineal
# Se usa un término de regularización para mejorar la estabilidad de los cálculos
def regresionLinealCholesky(X, y, alpha=1e-5):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    #verficamos si el numero de filas de x e y son iguales
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Error: el número de filas en X ({X.shape[0]}) debe ser igual al tamaño de y ({y.shape[0]}).")
    # Calcular X^T X y X^T y
    xtX = X.T @ X
    xtY = X.T @ y
    # Añadir término de regularización a la diagonal
    xtX_reg = xtX + alpha * np.eye(xtX.shape[0])
    # Verifica que la matriz regularizada xtX_reg sea cuadrada, simétrica y definida positiva
    verificarSimetriaPositiva(xtX_reg)
    #para obtener la matriz triangular inferios l
    L = descomposicionCholesky(xtX_reg)
    #resolver sistema
    beta = resolverSistemaCholesky(L, xtY)

    return beta


