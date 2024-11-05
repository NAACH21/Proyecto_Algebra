def verificarSimetriaPositiva(matriz):
    # comprobando la matriz cuadrada
    n = len(matriz)
    for i in range(n):
        if len(matriz[i]) != n:
            raise ValueError("Error:la  matriz X no es cuadrada")

    # comprobando qque sea simetrica
    print("hol")
    for i in range(n):
        for j in range(i, n):
            if matriz[i][j] != matriz[j][i]:
                raise ValueError("Error:la matriz X no es simetrica")

    # comprobando que sea  definida positiva
    for i in range(n):
        if matriz[i][i] <= 0:
            raise ValueError("Error:la matriz X no es definida positiva.")

    return True

def descomposicionCholesky(matriz):
    if verificarSimetriaPositiva(matriz):
        n = len(matriz)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                suma = 0
                for k in range(j):
                    producto = L[i][k] * L[j][k]
                    suma = suma + producto

                if i == j:
                    valor = matriz[i][i] - suma
                    if valor <= 0:
                        raise ValueError("Error:la matriz no "
                                         "es definida positiva "
                                         "durante la descomposición")
                    L[i][j] = valor ** 0.5
                else:
                    if L[j][j] == 0:
                        raise ValueError(
                            "Error:división por 0 "
                            "durante el calculo de la "
                            "descomposición")
                    L[i][j] = (matriz[i][j] - suma) / L[j][j]

        return L

def resolverSistemaCholesky(L, b):
    n = len(L)
    if len(b) != n:
        raise ValueError("Error:las dimensiones de L y b no coinciden.")
    #  L * y = b
    y = [0] * n
    for i in range(n):
        suma = 0
        for k in range(i):
            product = L[i][k] * y[k]
            suma = suma + product
        y[i] = (b[i] - suma) / L[i][i]

    #  L.T * x = y
    x = [0] * n
    for i in range(n - 1, -1, -1):
        suma = 0
        for k in range(i + 1, n):
            product = L[k][i] * x[k]
            suma = suma + product
        x[i] = (y[i] - suma) / L[i][i]

    return x


def regresionLinealCholesky(X, y):
    verificarSimetriaPositiva(X)
    if len(X) < 2 or len(X[0]) < 1:
        raise ValueError("Error:la matriz X debe tener al menos dos filas y una columna.")
    if len(X) != len(y):
        raise ValueError(f"Error:el número de filas en X ({len(X)}) debe se igual con el tamaño de y ({len(y)}).")
    #  X^T * X
    n = len(X)
    p = len(X[0])
    xtX = [[0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            suma = 0
            for k in range(n):
                producto = X[k][i] * X[k][j]
                suma = suma + producto
            xtX[i][j] = suma
    #  X^T * y
    xtY = [0] * p
    for i in range(p):
        suma = 0
        for k in range(n):
            producto = X[k][i] * y[k]
            suma = suma + producto
        xtY[i] = suma
     # verificando que xtX sea cuadrada y definida positiva
    verificarSimetriaPositiva(xtX)
    #  X^T * X en L * L.T
    L = descomposicionCholesky(xtX)
    #  L * L.T * beta = X^T * y
    beta = resolverSistemaCholesky(L, xtY)

    return beta


