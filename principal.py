
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from Cholesky import *


if __name__ == '__main__':

    # Configurar semilla
    np.random.seed(42)

    # Definir parámetros
    n_observations = 100
    n_variables = 5
    beta_true = np.array([3.0, 2.5, -1.2, 3.3, -0.7, 1.5])  # Intercepto + coeficientes

    # Generar variables independientes
    X = np.random.normal(0, 1, (n_observations, n_variables))

    # Agregar intercepto
    X_with_intercept = np.hstack((np.ones((n_observations, 1)), X))

    # Generar ruido
    epsilon = np.random.normal(0, 0.5, n_observations)

    # Calcular Y
    Y = X_with_intercept @ beta_true + epsilon

    # Crear DataFrame
    variable_names = ['Intercepto'] + [f'X{i + 1}' for i in range(n_variables)]
    data = pd.DataFrame(X_with_intercept, columns=variable_names)
    data['Y'] = Y

    # Preparar datos para el modelo
    X_model = data[variable_names]
    Y_model = data['Y']

    # Ajustar modelo con statsmodels
    model = sm.OLS(Y_model, X_model).fit()
    print(model.summary())

    # Predicciones
    Y_pred = model.predict(X_model)

    # Resumen del modelo
    print(model.summary())

    # Visualizar la relación entre Y real y Y estimado
    plt.scatter(Y_model, Y_pred)
    plt.xlabel('Y Real')
    plt.ylabel('Y Estimado')
    plt.title('Relación entre Y Real y Y Estimado')
    plt.grid(True)
    plt.show()
    print("====================================================")
    # Agregar una columna de unos a X

    # X_with_intercept = add_intercept_column(X_standardized[:, :3])  # Usamos solo las 3 primeras variables

    try:
        beta2 = regresionLinealCholesky(X_model, Y_model)
        print(beta2)
        print("EEE")
        ms1, r22 = calculate_errors(X_model, Y_model, beta2)
        print("Error cuadrático medio (MSE):", ms1)
        print("Coeficiente de determinación (R^2):", r22)
        print("Coeficientes beta :", beta2)
        print_regression_equation(beta2, include_intercept=True)
    except ValueError as e:
        print("Error:", e)

