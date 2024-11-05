import numpy as np

def svd_linear_regression_regularized(X, y, alpha=1e-10):
    # Agrega una columna de unos para el término independiente
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Aplica regularización
    S_reg = S / (S ** 2 + alpha)  # Regularización de los valores singulares

    # Calcula los coeficientes
    beta = Vt.T @ np.diag(S_reg) @ U.T @ y

    return beta


if __name__ == '__main__':
    # Ejemplo de uso
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([3, 5, 7, 9])

    coeficientes = svd_linear_regression_regularized(X, y)
    print("Coeficientes de regresión (SVD Regularizado):", coeficientes)