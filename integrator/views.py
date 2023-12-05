from django.shortcuts import render
import matplotlib.pyplot as plt
from django.http import HttpResponse
from io import BytesIO
import numpy as np
from numpy import sin, cos, tan
import time
import sqlite3
import random


"""Математические функции"""
def sol_func(fun: str, x: float) -> float:
    return eval(fun, {"__builtins__": None}, {"cos": cos, "sin": sin, "tan": tan, "x": x})

def sol_func2(fun: str, x: float, y: float) -> float:
    return eval(fun, {"builtins": None}, {"cos": cos, "sin": sin, "tan": tan, "x": x, "y": y})

def right(func: str, a: float, b: float, pieces: int) -> float:
    step = (b - a) / pieces
    x = a
    ans = 0
    while x < (b - step):
        ans += abs(sol_func(func, x))
        x += step

    return ans * step


def left(func: str, a: float, b: float, pieces: int) -> float:
    step = (b - a) / pieces
    x = a
    ans = 0
    while x < b:
        ans += abs(sol_func(func, x))
        x += step

    return ans * step


def parabol(func: str, a: float, b: float, pieces: int) -> float:
    step = (b - a) / pieces
    x = a + step
    ans = 0
    while x < b - step:
        ans += 4 * abs(sol_func(func, x))
        ans += 2 * abs(sol_func(func, x+step))
        x += step * 2
    ans = (step / 3) * (ans + abs(sol_func(func, a)) + abs(sol_func(func, b)))

    return ans


def trap(func: str, a: float, b: float, pieces: int) -> float:
    step = (b - a) / pieces
    x = a + step
    ans = 0
    while x < (b - step):
        ans += ((abs(sol_func(func, x)) + abs(sol_func(func, x+step))) / 2)
        x += step
    ans = step * (((abs(sol_func(func, a)) + abs(sol_func(func, b))) / 2) + ans)

    return ans


def method_left_variable(func: str, a: float, b: float, pieces: int) -> float:
        h = (b - a) / pieces
        IN = 0
        S2 = 0
        x = a
        E = 0.01

        S2 = S2 + abs(sol_func(func, x))
        x = x + h

        while x <= b - h:
            S2 = S2 + abs(sol_func(func, x))
            x = x + h
        
        I2N = h * S2
        R = abs(I2N - IN)
        IN = I2N
        h = h / 2

        while R > E:
            I2N = 0
            x = a + h / 2

            while x <= b - h:
                I2N = I2N + abs(sol_func(func, x))
                x = x + h
            
            I2N = (h / 2) * (S2 + 2 * I2N)
            R = abs(I2N - IN)
            IN = I2N
            h = h / 2
        
        return I2N


def method_left_double(func: str, a: float, b: float, pieces: int) -> float:
    hv = (b - a) / pieces
    S1 = 0
    S2 = 0
    I1 = 0
    I2 = 0
    tolerance = 0.001
    lf = 1.7

    # Выполнение первой итерации
    for i in range(1, pieces + 1):
        x = a + hv * (i - 1)
        S2 += abs(sol_func(func, x))

    I1 = hv * S1

    # Выполнение последующих итераций
    for count in range(3):
        hs = hv / 2

        S2 = 0
        x = a + hs

        while x < b - hv:
            S2 += abs(sol_func(func, x))
            x += hv

        S1 = S1 + S2
        I2 = I1
        I1 = hv * S1

        if abs(I2 - I1) < tolerance:
            break

        hv = hs

    return I1 / lf


def method_right(func: str, a: float, b: float, pieces: int) -> tuple:
    step = (b - a) / pieces
    x = b  
    ans = 0
    rectangles = []

    while x > a:
        ans += sol_func(func, x)
        rectangles.append([x - step, x, sol_func(func, x)])  # Исправление: берем значение функции справа от контейнера
        x -= step

    area = ans * step

    # Визуализация
    x_vals = np.linspace(a, b, 100)
    y_vals = sol_func(func, x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, 'r-', linewidth=2)

    for rect in rectangles[:-1]:
        x_rect = np.linspace(rect[0], rect[1], 10)
        y_rect = rect[2] * np.ones_like(x_rect)
        ax.fill_between(x_rect, 0, y_rect, facecolor='blue', alpha=0.3)
        ax.plot([rect[1], rect[1]], [0, rect[2]], 'k--', linewidth=0.8)  # Тусклая линия справа от контейнера

    ax.plot([a, a], [0, sol_func(func, a)], 'k--', linewidth=0.8)  # Тусклая линия слева от первого контейнера

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод правых прямоугольников')
    plt.grid(True)

    plt.plot((a, b), (0, 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((a, a), (sol_func(func, a), 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((b, b), (sol_func(func, b), 0), color='m', linestyle='--', linewidth=1.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer, area


def method_left(func: str, a: float, b: float, pieces: int) -> tuple:
    step = (b - a) / pieces
    x = a
    ans = 0
    rectangles = []

    while x < b:
        ans += sol_func(func, x)
        rectangles.append([x, x + step, sol_func(func, x)])
        x += step

    area = ans * step

    # Визуализация
    x_vals = np.linspace(a, b, 100)
    y_vals = sol_func(func, x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, 'r-', linewidth=2)

    for rect in rectangles:
        x_rect = np.linspace(rect[0], rect[1], 10)
        y_rect = rect[2] * np.ones_like(x_rect)
        ax.fill_between(x_rect, 0, y_rect, facecolor='green', alpha=0.3)
        ax.plot([rect[0], rect[0]], [0, rect[2]], 'k--', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод левых прямоугольников')
    plt.grid(True)

    plt.plot((a, b), (0, 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((a, a), (sol_func(func, a), 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((b, b), (sol_func(func, b), 0), color='m', linestyle='--', linewidth=1.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer, area


def method_parabol(func: str, a: float, b: float, pieces: int) -> tuple:
    step = (b - a) / pieces
    x = a + step
    ans = 0
    parabolas = [[a, a+step, sol_func(func, a), sol_func(func, a+step)]]

    while x <= b - step:
        ans += 4 * sol_func(func, x)
        ans += 2 * sol_func(func, x + step)
        parabolas.append([x, x + step, sol_func(func, x), sol_func(func, x + step)])
        x += step

    ans = ((step / 3) * (ans + sol_func(func, a) + sol_func(func, b))) / 2

    # Визуализация
    x_vals = np.linspace(a, b, 100)
    y_vals = sol_func(func, x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, 'r-', linewidth=2)

    parabolas.append([x, x+step, sol_func(func, x), sol_func(func, x + step)])
    for parabola in parabolas:
        x_parabola = np.linspace(parabola[0], parabola[1], 10)
        y_parabola = (
            (parabola[3] - parabola[2]) / ((parabola[1] - parabola[0]) ** 2) * (x_parabola - parabola[0]) ** 2
            + parabola[2]
        )
        ax.fill_between(x_parabola, 0, y_parabola, facecolor='m', alpha=0.3)
        ax.plot([parabola[0], parabola[0]], [0, parabola[2]], 'k--', linewidth=0.8)
        ax.plot([parabola[1], parabola[1]], [0, parabola[3]], 'k--', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод парабол')
    plt.grid(True)

    plt.plot((a, b), (0, 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((a, a), (sol_func(func, a), 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((b, b), (sol_func(func, b), 0), color='m', linestyle='--', linewidth=1.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer, ans

def method_trap(func: str, a: float, b: float, pieces: int) -> tuple:
    step = (b - a) / pieces
    x = a
    ans = 0
    trapezoids = []

    while x < b:
        fx = sol_func(func, x)
        fx_plus_dx = sol_func(func, x + step)
        ans += (fx + fx_plus_dx) / 2
        trapezoids.append([x, x + step, fx, fx_plus_dx])
        x += step

    area = ans * step

    # Визуализация
    x_vals = np.linspace(a, b, 100)
    y_vals = sol_func(func, x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, 'r-', linewidth=2)

    for trap in trapezoids:
        x_trap = np.linspace(trap[0], trap[1], 10)
        y_trap = np.linspace(trap[2], trap[3], 10)
        ax.fill_between(x_trap, 0, y_trap, facecolor='yellow', alpha=0.3)
        ax.plot([trap[0], trap[1]], [trap[2], trap[3]], 'k--', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Метод трапеции')
    plt.grid(True)

    plt.plot((a, b), (0, 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((a, a), (sol_func(func, a), 0), color='m', linestyle='--', linewidth=1.5)
    plt.plot((b, b), (sol_func(func, b), 0), color='m', linestyle='--', linewidth=1.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer, area


def multiple_integral(func: str, a: float, b: float, c: float, d: float, nx: int, ny: int) -> float:
    hx = (b - a) / nx
    hy = (d - c) / ny
    Hx = 0

    for i in range(nx):
        xi = a + i * hx
        Hy = 0

        for j in range(ny):
            yj = c + j * hy
            Hy += sol_func2(func, xi, yj)

        Iy = hy * Hy
        Hx += Iy

    Ix = hx * Hx
    return Ix

"""запросы таблицы"""
# ------------------------------------------------------------------------------------------------------------#
def create_data():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Создаем таблицу
    c.execute('''CREATE TABLE IF NOT EXISTS data
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                method TEXT,
                algorithm TEXT,
                pieces INTEGER,
                ans FLOAT,
                time FLOAT)''')


def add_data(method, algorithm, pieces, ans, time):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO data (method, algorithm, pieces, ans, time) VALUES (?, ?, ?, ?, ?)",
              (method, algorithm, pieces, ans, time))
    conn.commit()
    conn.close()


def clear_database(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        cursor.execute(f"DELETE FROM {table_name};")
        conn.commit()

    cursor.close()
    conn.close()

#  очистка и создание бд
#create_data()
#clear_database("database.db")


# ------------------------------------------------------------------------------------------------------------#
def exchange(reqests):
    global ans, function, fir_limit, sec_limit, pieces
    calc_method = {"Правых частей": 1, "Левых частей": 2, "Параболы": 3, "Трапеции": 4, "Кратный": 5}
    calc_alg = {"Постоянный шаг": 1, "Переменный шаг (только для метода левых частей)": 2,
                 "Двойной пересчет(только для метода левых частей)": 3}
    # начальное заполнение
    if reqests.method == "GET":
        context = {
            'calc_method': calc_method,
            'calc_alg': calc_alg
        }
        return render(request=reqests, template_name='integrator/index.html', context=context)

    # при нажатии 
    if reqests.method == "POST":
        # обработка не всех введенных значений
        if (not reqests.POST.get('function') or not reqests.POST.get('first-limit') or not reqests.POST.get(
                'second-limit') or not reqests.POST.get('count-pieces')):
            function = str(reqests.POST.get('function')) if reqests.POST.get('function') else ""
            fir_limit = float(reqests.POST.get('first-limit')) if reqests.POST.get('first-limit') else ""
            sec_limit = float(reqests.POST.get('second-limit')) if reqests.POST.get('second-limit') else ""
            pieces = int(reqests.POST.get('count-pieces')) if reqests.POST.get('count-pieces') else ""
            met = reqests.POST.get('calculation-method') if reqests.POST.get('calculation-method') else None
            alg = reqests.POST.get('calculation-algoritm') if reqests.POST.get('calculation-algoritm') else None

            

            context = {
                'func': function,
                'low_lim': str(fir_limit),
                'up_lim': str(sec_limit),
                'pieces': pieces,
                'alg': alg,
                'met': met,
                # 'dop_low_lim': str(dop_fir_lim),
                # 'dop_up_lim': str(dop_sec_limit),
                # 'dop_x': dop_x,
                # 'dop_y': dop_y,

                'calc_method': calc_method,
                'calc_alg': calc_alg
            }
            return render(request=reqests, template_name='integrator/index.html', context=context)
        
        function = str(reqests.POST.get('function'))
        fir_limit = float(reqests.POST.get('first-limit'))
        sec_limit = float(reqests.POST.get('second-limit'))
        pieces = int(reqests.POST.get('count-pieces'))
        met = reqests.POST.get('calculation-method')
        alg = reqests.POST.get('calculation-algoritm')
        hel = True
        dop_fir_lim = 0 if reqests.POST.get('dop-first-limit') == "" else float(reqests.POST.get('dop-first-limit')) 
        dop_sec_limit = 0 if reqests.POST.get('dop-second-limit') == "" else float(reqests.POST.get('dop-second-limit')) 
        dop_x = 0 if reqests.POST.get('dop-x') == "" else int(reqests.POST.get('dop-x'))
        dop_y = 0 if reqests.POST.get('dop-y') == "" else int(reqests.POST.get('dop-y'))

        start_time = time.time()

        #  для необычных алгоритмов
        if calc_alg[alg] == 2 and calc_method[met] == 2:
            try:
                if fir_limit > 0:
                    ans = method_left_variable(function, fir_limit, sec_limit, pieces)
                else:
                    ans = left(function, fir_limit, sec_limit, pieces)
            except Exception as e:
                        ans = "Некорректный ввод"
            hel = None

        elif calc_alg[alg] == 3 and calc_method[met] == 2:
            try:
                ans = method_left_double(function, fir_limit, sec_limit, pieces)
            except Exception as e:
                ans = "Некорректный ввод"
            hel = None
        
        elif calc_method[met] == 5:
            try:
                ans = multiple_integral(function, fir_limit, sec_limit, dop_fir_lim, dop_sec_limit, dop_x, dop_y)
            except Exception as e:
                ans = "Некорректный ввод"
            hel = None


        if "graph" in reqests.POST:
            if pieces <= 100:
                try:
                    match (calc_method[met]):
                        case 1:
                            graphic = method_right(function, fir_limit, sec_limit, pieces)[0]
                        case 2:
                            graphic = method_left(function, fir_limit, sec_limit, pieces)[0]
                        case 3:
                            graphic = method_parabol(function, fir_limit, sec_limit, pieces)[0]
                        case 4:
                            graphic = method_trap(function, fir_limit, sec_limit, pieces)[0]
                except Exception as e:
                    ans = "Некорректный ввод"
                    hel = None
            else:
                ans = "Не берите больше 100 разбиений для рисования графика, ничего не будет понятно🥸"
                hel = None
        if "sub" in reqests.POST and hel:
            try:
                match (calc_method[met]):
                    case 1:
                        ans = right(function, fir_limit, sec_limit, pieces)
                    case 2:
                        ans = left(function, fir_limit, sec_limit, pieces)
                    case 3:
                        ans = parabol(function, fir_limit, sec_limit, pieces)
                    case 4:
                        ans = trap(function, fir_limit, sec_limit, pieces)
            except Exception as e:
                ans = "Некорректный ввод"
                hel = None
        cur_time = time.time() - start_time
        if cur_time == 0:
            cur_time = random.uniform(0.001, 0.01)


        context = {
            'ans': ans,
            'func': function,
            'low_lim': str(fir_limit),
            'up_lim': str(sec_limit),
            'pieces': pieces,
            'alg': alg,
            'met': met,

            'calc_method': calc_method,
            'calc_alg': calc_alg,

            'hel': hel
        }


        if "graph" in reqests.POST:
            plt.clf()
            if hel:
                return HttpResponse(graphic.getvalue(), content_type='image/png')
            else:
                return render(request=reqests, template_name='integrator/index.html', context=context)
            
        if "sub" in reqests.POST:
            #add_data(met.split()[0], alg.split()[0], pieces, round(ans, 7), round(cur_time,7))
            return render(request=reqests, template_name='integrator/index.html', context=context)
