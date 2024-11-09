from DSV import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Fijar una semilla para la reproducibilidad
    np.random.seed(0)

    # Generar 100 observaciones para 5 variables independientes (regresores)
    X1 = np.random.normal(0, 1, 100)
    X2 = np.random.normal(0, 1, 100)
    X3 = np.random.normal(0, 1, 100)
    X4 = np.random.normal(0, 1, 100)
    X5 = np.random.normal(0, 1, 100)

    # Definir los coeficientes reales
    beta_0 = 2
    beta_1 = 1.5
    beta_2 = -2
    beta_3 = 0.5
    beta_4 = 3
    beta_5 = -1

    # Generar el término de error
    epsilon = np.random.normal(0, 1, 100)

    # Calcular la variable dependiente Y
    Y = beta_0 + beta_1 * X1 + beta_2 * X2 + beta_3 * X3 + beta_4 * X4 + beta_5 * X5 + epsilon
    y_2 = Y
    # Crear un DataFrame con las variables independientes y dependiente
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'Y': Y})

    # Ajustar el modelo de regresión lineal
    X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
    x_2=X
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = sm.add_constant(X)  # Añadir una constante para el intercepto
    model = sm.OLS(data['Y'], X).fit()

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
    print("====================================================")
    # Agregar una columna de unos a X
    #X_with_intercept = add_intercept_column2(x_2)

    #X_with_intercept = add_intercept_column(X_standardized[:, :3])  # Usamos solo las 3 primeras variables

    try:
        beta = regression_coefficients_svd2(x_2, y_2)
        mse, r2 = calculate_errors2(x_2, y_2, beta)
        print("Coeficientes de regresión (incluyendo intercepto):", beta)
        print("Error cuadrático medio (MSE):", mse)
        print("Coeficiente de determinación (R^2):", r2)
        print_regression_equation2(beta, include_intercept=True)
    except ValueError as e:
        print("Error:", e)