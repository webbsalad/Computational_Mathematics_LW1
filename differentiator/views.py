from django.shortcuts import render
import numpy as np
import sympy as sp
from numpy import sin, cos, tan
from django.shortcuts import render
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from django.http import HttpResponse


def sol_func(eq: str, x: float, y: float=None, z: float=None) -> float:
    context = {"__builtins__": None, "cos": cos, "sin": sin, "tan": tan, "x": x, "y": y}
    return eval(eq, context)

def find_derivative(func: str, var: str, var2):
    var = sp.symbols(str(var))
    expr = sp.sympify(func)
    ans = sp.diff(expr, var)

    return str(ans)

def eiler(func: str, a: float, b: float, pieces: int) -> tuple[list[float]]:
    h = (b - a) / pieces
    x = np.zeros(pieces+1)
    y = np.zeros(pieces+1)
    x[0] = a
    y[0] = b

    for i in range(pieces):
        y[i+1] = y[i] + h*sol_func(func, x[i], y[i])
        x[i+1] = x[i] + h

    return x, y

def rune(func: str, a: float, b: float, pieces: int) -> tuple[list[float]]:
    h = (b - a) / pieces
    x = np.zeros(pieces+1)
    y = np.zeros(pieces+1)
    x[0] = a
    y[0] = b

    for i in range(pieces):
        k1 = h*sol_func(func, x[i], y[i])
        k2 = h*sol_func(func, x[i] + h/2, y[i] + k1/2)
        k3 = h*sol_func(func, x[i] + h/2, y[i] + k2/2)
        k4 = h*sol_func(func, x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        x[i+1] = x[i] + h

    return x, y

def double_eiler() -> list[list[float]]:
    a = 1
    b = 1.5
    n = 10
    x = a
    y = 0.77
    z = -0.44
    y0 = y
    h = 0.1
    x += h
    ans = [[], [], []]
    while x < b + h:
      y = y + h * z
      z = z - h * (z/x + y0)
      ans[0].append(x); ans[1].append(y); ans[2].append(z)
      x = x + h
      y0 = y
      
    return ans

def system(h, t_start, t_end, x0, y0, z0):
    num_steps = int((t_end - t_start) / h) + 1

    t = np.linspace(t_start, t_end, num_steps)
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(num_steps - 1):
        dx_dt = -2 * x[i] + 5 * z[i]
        dy_dt = np.sin(t[i] - 1) * x[i] - y[i] + 3 * z[i]
        dz_dt = -x[i] + 2 * z[i]

        x[i + 1] = x[i] + h * dx_dt
        y[i + 1] = y[i] + h * dy_dt
        z[i + 1] = z[i] + h * dz_dt

    return t, x, y, z

def print_graphic(func: str, a: float, b: float, pieces: int, method: int) -> tuple:
    fig, ax = plt.subplots()
    match method:
        case 1:
            x_vals, y_vals = eiler(func, a, b, pieces)
        case 2:
            x_vals, y_vals = rune(func, a, b, pieces)
            

    ax.plot(x_vals, y_vals)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод Эйлера')
    plt.grid(True)


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def  snag(reqests):
    calc_method = {"Эйлера": 1, "Рунге-Кутта": 2}
    
    if reqests.method == "GET":
        context = {
            'calc_method': calc_method,
        }
    
        return render(request=reqests, template_name='differentiator/fir.html', context=context)
    
    if reqests.method == "POST":
        function = str(reqests.POST.get('function')) if reqests.POST.get('function') else ""
        fir_limit = float(reqests.POST.get('first-limit')) if reqests.POST.get('first-limit') else ""
        sec_limit = float(reqests.POST.get('second-limit')) if reqests.POST.get('second-limit') else ""
        pieces = int(reqests.POST.get('count-pieces')) if reqests.POST.get('count-pieces') else ""
        met = reqests.POST.get('calculation-method') if reqests.POST.get('calculation-method') else None
        req = True

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
            return render(request=reqests, template_name='differentiator/fir.html', context=context)
        return context, function, fir_limit, sec_limit, pieces, met, req


def menu(reqests):
    return render(request=reqests, template_name='differentiator/menu.html')
    
def fir(reqests):
    calc_method = {"Эйлера": 1, "Рунге-Кутта": 2}

    if reqests.method == "GET":
        context = {
            'calc_method': calc_method,
        }
    
        return render(request=reqests, template_name='differentiator/fir.html', context=context)
    
    if reqests.method == "POST":

        context, function, fir_limit, sec_limit, pieces, met, req =  snag(reqests)

        # при кнопке рассчитать
        if "sub" in reqests.POST or "copy" in reqests.POST:
            try:
                match (calc_method[met]):
                    case 1:
                        cur = eiler(function, fir_limit, sec_limit, pieces)
                    case 2:
                        cur = rune(function, fir_limit, sec_limit, pieces)

                ans = f"X: {cur[0]} \n Y: {cur[1]}"

                
            except Exception as e:
                ans = "Некорректный ввод"
            context['ans'] = ans
            return render(request=reqests, template_name='differentiator/fir.html', context=context)
        

        if "graph" in reqests.POST:
            try:
                graphic = print_graphic(function, fir_limit, sec_limit, pieces, calc_method[met])
                plt.clf()
                return HttpResponse(graphic.getvalue(), content_type='image/png')
            
            except Exception as e:
                context['ans'] = "Некорректные данные"
                return render(request=reqests, template_name='differentiator/fir.html', context=context)
        
def sec(reqests):
    context = {

    }
    if reqests.method == "GET":    
        return render(request=reqests, template_name='differentiator/sec.html', context=context)
    
    if reqests.method == "POST":

        if "sub" in reqests.POST or "copy" in reqests.POST:
            try:
                cur = double_eiler()
                ans = f"X: {cur[0]} \n Y: {cur[1]} \n Z: {cur[2]}"
            except Exception as e:
                ans = "Некорректные данные"

            context['ans'] = ans
            return render(request=reqests, template_name='differentiator/sec.html', context=context)

def sis(reqests):
    context = {
        
    }
    if reqests.method == "GET":
        return render(request=reqests, template_name='differentiator/sis.html', context=context)
    
    if reqests.method == "POST":

        if "sub" in reqests.POST or "copy" in reqests.POST:
            cur = []
            try:
                cur = system(0.03, 0, 0.3, 2, 1, 1)
                ans = f"Xt: {cur[0]} \nYt: {cur[1]} \nZt{cur[2]}"
            except Exception as e:
                ans = "Некорректные данные"

            context['ans'] = ans
            return render(request=reqests, template_name='differentiator/sis.html', context=context)
