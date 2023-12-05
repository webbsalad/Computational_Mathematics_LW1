import sympy as sp

def sol_func(expression, variable, x):
    return eval(expression, {variable: x})

def newton_method(expression, variable, a, b, tol=1e-6, max_iter=100):
    """
    Метод Ньютона для нахождения корня уравнения f(x) = 0 в интервале [a, b].

    Параметры:
    - expression: Строка, содержащая выражение f(x).
    - variable: Переменная в выражении.
    - a: Левая граница интервала.
    - b: Правая граница интервала.
    - tol: Допустимая погрешность.
    - max_iter: Максимальное количество итераций.

    Возвращает:
    - Корень уравнения (приближенный).
    - Количество итераций.
    """
    x = (a + b) / 2  # Начальное приближение

    f_expr = expression
    df_expr = diff(expression, variable)

    f = lambda x: sol_func(f_expr, variable, x)
    df = lambda x: sol_func(df_expr, variable, x)

    for i in range(max_iter):
        fx = f(x)

        if abs(fx) < tol:
            return x, i

        dfx = df(x)
        if dfx == 0:
            raise ValueError("Производная равна нулю. Метод Ньютона не сходится.")

        x = x - fx / dfx

        if x < a or x > b:
            raise ValueError("Получено значение вне интервала [a, b]. Метод Ньютона не сходится.")

    raise ValueError("Достигнуто максимальное количество итераций. Метод Ньютона не сходится.")

def find_derivative(func: str, var: str, var2: str):
    var = sp.symbols(str(var))
    var2 = sp.symbols(str(var2))
    expr = sp.sympify(func)
    ans = sp.diff(expr, var, var2)

    return str(ans)


print(find_derivative("y*(1-x)", 'x', 'y'))