from django.shortcuts import render
import numpy as np
from numpy import sin, cos, tan
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import HttpResponse
import sympy as sp


def sol_func(eq: str, x: float, y: float = None) -> float:
    context = {"__builtins__": None, "cos": cos, "sin": sin, "tan": tan, "x": x, "y": y}
    return eval(eq, context)

def find_derivative(func: str) -> str:
    x = sp.symbols('x')
    func_expr = sp.sympify(func)
    derivative_expr = sp.diff(func_expr, x)
    derivative = str(derivative_expr)

    return derivative

def chebushev_ssin():
    def chebyshev(x, n):
        if n== 0:
            return np.ones_like(x)
        elif n ==1:
            return x
        else:
            return 2 * x * chebyshev(x, n-1) - chebyshev(x, n-2)
    
    def chebyshev_sin(x, n, a):
        t = chebyshev(x, n)
        return np.dot(a, t)


    a = np.array([0.9999998, 1.0000000, 0.5000063, 0.1666674, 0.0416350, 0.0083298, 0.0014393, 0.0002040])
    b = 2 * 10**(-7)
    x = 0.5
    if abs(x) <= 1:
        result = np.sin(x) * (1 + b * chebyshev_sin(x, 7, a))
        return(str(result))
    else:
        return("Некорректный ввод")

def cheb(alg):
    match(alg):
        case 1:
            def chebyshev(x, n):
                if n== 0:
                    return np.ones_like(x)
                elif n ==1:
                    return x
                else:
                    return 2 * x * chebyshev(x, n-1) - chebyshev(x, n-2)
            
            def chebyshev_exp(x, n, a):
                t = chebyshev(x, n)
                return np.dot(a, t)


            a = np.array([0.9999998, 1.0000000, 0.5000063, 0.1666674, 0.0416350, 0.0083298, 0.0014393, 0.0002040])
            b = 2 * 10**(-7)
            x = 0.5
            if abs(x) <= 1:
                result = np.exp(x) * (1 + b * chebyshev_exp(x, 7, a))
                return(str(result))
            else:
                return("Некорректный ввод")
        
        case 2:
            def chebyshev(x, n):
                if n== 0:
                    return np.ones_like(x)
                elif n ==1:
                    return x
                else:
                    return 2 * x * chebyshev(x, n-1) - chebyshev(x, n-2)
            
            def chebyshev_sin(x, n, a):
                t = chebyshev(x, n)
                return np.dot(a, t)


            a = np.array([0.9999998, 1.0000000, 0.5000063, 0.1666674, 0.0416350, 0.0083298, 0.0014393, 0.0002040])
            b = 2 * 10**(-7)
            x = 0.5
            if abs(x) <= 1:
                result = np.sin(x) * (1 + b * chebyshev_sin(x, 7, a))
                return(str(result))
            else:
                return("Некорректный ввод")

def iter(alg):
    match(alg):
        case 1:
            def sqrt_iteration(x, y0, epsilon=1e-6, max_iterations=100):
                yi = y0
                for i in range(max_iterations):
                    yi_plus_1 = 0.5 * (yi + x / yi)
                    if abs(yi_plus_1 - yi) < epsilon:
                        return yi_plus_1
                    yi = yi_plus_1
                return yi

            # Вычисление значения элементарной функции √𝑥 для заданных значений x
            x_values = [14.76, 0.142]
            y0_values = [3.8, 0.4]

            arr = []
            for x, y0 in zip(x_values, y0_values):
                arr.append(sqrt_iteration(x, y0))
            return arr

        case 2:
            # Функция для вычисления значения элементарной функции 1/√𝑥
            def inverse_sqrt_iteration(x, y0, epsilon=1e-6, max_iterations=100):
                yi = y0
                for i in range(max_iterations):
                    yi_plus_1 = (yi / 2) * (3 - x * yi**2)
                    if abs(yi_plus_1 - yi) < epsilon:
                        return yi_plus_1
                    yi = yi_plus_1
                return yi

            # Вычисление значения элементарной функции 1/√𝑥 для заданных значений x
            x_values = [17.32, 0.464]
            y0_values = [0.24, 1.5]

            arr = []
            for x, y0 in zip(x_values, y0_values):
                arr.append(inverse_sqrt_iteration(x, y0))
            return arr


def exchange(reqests):
    calc_method = {"Чебышева": 1, "Итераций": 2}
    calc_alg = {"Экспонента(√x)": 1, "Синус(1/√x)": 2}

    if reqests.method == "GET":
        context = {
            'calc_method': calc_method,
            'calc_alg': calc_alg
        }
    
        return render(request=reqests, template_name='elementary/index.html', context=context)
    
    if reqests.method == "POST":
        met = reqests.POST.get('calculation-method') if reqests.POST.get('calculation-method') else None
        alg = reqests.POST.get('calculation-algoritm') if reqests.POST.get('calculation-algoritm') else None
        
        context = {
                'met': met,
                'alg': alg,
                'calc_method': calc_method,
                'calc_alg': calc_alg
            }
    
        
        if "sub" in reqests.POST:
            
            cur = []
            try:
                match (calc_method[met]):
                    case 1:
                        cur = cheb(calc_alg[alg])
                    case 2:
                        cur = iter(calc_alg[alg])
                ans = f"Массив решений: {cur}"
            except Exception as e:
                ans = "Некорректный ввод"
            context['ans'] = ans
            return render(request=reqests, template_name='elementary/index.html', context=context)
        
            

#from numpy.polynomial.chebyshev import Chebyshev