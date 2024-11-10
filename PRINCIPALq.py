from GramSchmidt import *
from TriangulaciónHouseholder import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generar una nueva matriz aleatoria con más observaciones (50x10)
    np.random.seed(0)  # Para reproducibilidad
    X_base = np.random.rand(50, 10)

    # Estandarización de la matriz X_base
    X_standardized = (X_base - np.mean(X_base, axis=0)) / np.std(X_base, axis=0)
    X_standardized = np.hstack((np.ones((X_standardized.shape[0], 1)), X_standardized))

    # Agregar una columna de unos a X_base para el intercepto, creando X no estandarizado
    X = np.hstack((np.ones((X_base.shape[0], 1)), X_base))

    # Definir coeficientes reales
    beta_real = np.array([50, -25, 15, -10, 5, 20, 8, 30, -5, 12, 18])

    # Generar Y con una relación lineal definida con X y añadir ruido
    Y = np.dot(X, beta_real) + np.random.normal(0, 10, X.shape[0])

    # Crear el DataFrame para el modelo
    data = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    data['Y'] = Y

    # Ajustar el modelo de regresión con statsmodels usando X no estandarizado
    model = sm.OLS(Y, X).fit()
    # Resumen del modelo
    print(model.summary())

    # Visualizar la relación entre Y real y Y estimado
    Y_est = model.predict(X)
    plt.scatter(Y, Y_est)
    plt.xlabel('Y Real')
    plt.ylabel('Y Estimado')
    plt.title('Relación entre Y Real y Y Estimado')
    plt.grid(True)
    plt.show()
    print("=====================================Gram-Schmidt===================================================")

    # Intentar calcular los coeficientes usando Gram-Schmidt con la matriz estandarizada
    try:
        beta = regression_coefficients(X_standardized, Y)
        mse, r2 = calculate_errors(X_standardized, Y, beta)
        print("Coeficientes de regresión (incluyendo intercepto):", beta)
        print("Error cuadrático medio (MSE):", mse)
        print("Coeficiente de determinación (R^2):", r2)
        print_regression_equation(beta, include_intercept=True)
    except ValueError as e:
        print("Error:", e)
    print("=========================================QR=========================================================")

    try:
        # Calcular los coeficientes utilizando la descomposición QR
        beta_qr = qr_regression_coefficients(X_standardized, Y)
        mse_qr, r2_qr = calculate_errors(X_standardized, Y, beta_qr)
        print("Coeficientes de regresión (incluyendo intercepto, QR):", beta_qr)
        print("Error cuadrático medio (MSE, QR):", mse_qr)
        print("Coeficiente de determinación (R^2, QR):", r2_qr)
        print_regression_equation(beta_qr, include_intercept=True)
    except ValueError as e:
        print("Error:", e)