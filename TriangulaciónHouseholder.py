import numpy as np

def householder_regression(X, y):
    # Añadir una columna de unos a X para el término independiente
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Triangulación de Householder
    m, n = X.shape
    R = X.copy()
    Q = np.eye(m)

    for i in range(n):
        # Crear el vector de Householder
        x = R[i:, i]
        if len(x) == 0:
            break
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        u_norm = np.linalg.norm(x - e)
        if u_norm == 0:
            continue
        u = (x - e) / u_norm

        # Construir la matriz de Householder
        H = np.eye(m)
        H[i:, i:] -= 2.0 * np.outer(u, u)

        # Actualizar R y Q
        R = H @ R
        Q = Q @ H

    # Resolver el sistema R_reducido * beta = Q^T * y
    R_reducido = R[:n, :]
    Qt_y = Q.T @ y
    beta = np.linalg.lstsq(R_reducido, Qt_y, rcond=None)[0]

    return beta

# Ejemplo de uso
if __name__ == "__main__":
    # Matriz de variables independientes
    X = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]])

    # Vector de la variable dependiente
    y = np.array([6, 10, 14])

    # Calcular los coeficientes de la regresión lineal
    beta = householder_regression(X, y)

    # Mostrar los coeficientes y la ecuación de la regresión
    print("Coeficientes beta:", beta)
    print(f"Ecuación de regresión: y = {beta[0]} + {beta[1]}*x1 + {beta[2]}*x2 + {beta[3]}*x3")
