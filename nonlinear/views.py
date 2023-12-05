from django.shortcuts import render
import numpy as np
from numpy import sin, cos, tan
from django.shortcuts import render
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from django.http import HttpResponse
from sympy import symbols, Eq, solve


def sol_func(eq: str, x: float, y: float = None) -> float:
    context = {"__builtins__": None, "cos": cos, "sin": sin, "tan": tan, "x": x, "y": y}
    return eval(eq, context)


def dihotomia(func: str, a: float, b: float, pieces: int) -> list[float]:
    eps = 1 / pieces

    def helper(func=func, a=a, b=b, eps=eps) -> float:
        x0 = (a + b) / 2
        while abs(a - b) >= eps:
            if sol_func(func, a) * sol_func(func, x0) > 0:
                a = x0
            else:
                b = x0
            x0 = (a + b) / 2

        return x0

    roots = []

    x0 = a + eps
    while x0 <= b:
        if sol_func(func, a) * sol_func(func, x0) < 0:
            
            roots.append(helper(func, a, x0, eps))
        a = x0
        x0 += eps

    if len(roots) == 0:
        return "Нет корней или взято слишком мало разбиений"
    return roots


def print_graphic(func: str, a: float, b: float, pieces: int, method: int) -> tuple:
    fig, ax = plt.subplots()
    match method:
        case 1:
            cur = dihotomia(func, a, b, pieces)
        
    if cur == "Нет корней или взято слишком мало разбиений":
        return False
    x_vals = cur
    x_points = np.arange(a, b, 0.1)
    y_points = [sol_func(func, i) for i in x_points]
    ax.plot(x_points, y_points)
    dop = (b - a) / 10
    ax.plot([a-dop, b+dop], [0, 0], color='black')

    for char in x_vals:
        ax.scatter(char, 0, color='black')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод Эйлера')
    plt.grid(True)


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def exchange(reqests):
    calc_method = {"Деления отрезка": 1, "Касательных": 2}

    if reqests.method == "GET":
        context = {
            'calc_method': calc_method,
        }
    
        return render(request=reqests, template_name='nonlinear/index.html', context=context)
    
    if reqests.method == "POST":
        function = str(reqests.POST.get('function')) if reqests.POST.get('function') else ""
        fir_limit = float(reqests.POST.get('first-limit')) if reqests.POST.get('first-limit') else ""
        sec_limit = float(reqests.POST.get('second-limit')) if reqests.POST.get('second-limit') else ""
        pieces = int(reqests.POST.get('count-pieces')) if reqests.POST.get('count-pieces') else ""
        met = reqests.POST.get('calculation-method') if reqests.POST.get('calculation-method') else None

        context = {
                'func': function,
                'low_lim': str(fir_limit),
                'up_lim': str(sec_limit),
                'pieces': pieces,
                'met': met,

                'calc_method': calc_method,
            }
        
        # обработка не всех введенных данных
        if (not reqests.POST.get('function') or not reqests.POST.get('first-limit') or not reqests.POST.get(
                'second-limit') or not reqests.POST.get('count-pieces')):
            return render(request=reqests, template_name='nonlinear/index.html', context=context)
        
        # при кнопке рассчитать
        if "sub" in reqests.POST or "copy" in reqests.POST:
            try:
                match (calc_method[met]):
                    case 1:
                        cur = dihotomia(function, fir_limit, sec_limit, pieces)
                ans = f"{cur}"
            except Exception as e:
                ans = "Некорректный ввод"
            context['ans'] = ans
            return render(request=reqests, template_name='nonlinear/index.html', context=context)
        
        if "graph" in reqests.POST:
            try:
                graphic = print_graphic(function, fir_limit, sec_limit, pieces, calc_method[met])
                plt.clf()
                if graphic:
                    return HttpResponse(graphic.getvalue(), content_type='image/png')
                else:
                    return render(request=reqests, template_name='nonlinear/index.html', context=context)
            
            except Exception as e:
                context['ans'] = "Некорректные данные"
                return render(request=reqests, template_name='nonlinear/index.html', context=context)