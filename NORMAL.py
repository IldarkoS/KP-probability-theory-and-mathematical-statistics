import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

EPS = np.array(
    [-0.15942282, -0.8410887, 2.15305708,
     -1.97161684,
     0.4172856,
     -0.08296028,
     0.20917913,
     -1.4237559,
     0.350999819,
     1.541904307,
     -0.618595332,
     0.803563112,
     1.019235419,
     0.13303219,
     -0.471654893,
     1.990168709,
     2.726648656,
     -1.356363182,
     3.068083913,
     0.274078655,
     -0.286778764,
     1.901556751,
     -0.558719215,
     1.349260321,
     -0.574000136,
     1.12601593,
     0.340159836,
     -1.413236767,
     0.015559032,
     0.783232046,
     -0.739046114,
     -3.720552269,
     -1.178010502,
     2.092098886,
     1.200454818,
     2.419930148,
     0.286940381,
     0.716142324,
     2.695037527,
     -2.414894773
     ])

n = 40
sigma = 2.6 ** 0.5
theta0 = 20
theta1 = -2
theta2 = -1
theta3 = -0.06

theta = np.array([theta0, theta1, theta2, theta3])


def y(th0, th1, th2, th3, t):
    return th3 * t ** 3 + th2 * t ** 2 + th1 * t + th0


def task_1():
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y = np.dot(X, theta) + EPS

    def get_stat(p, alpha=0.05):
        # Набор наблюдений для степени многочлена p
        X_m = X[:40, :p + 1]
        Theta_m = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_m), X_m)), np.transpose(X_m)), Y)

        quant_m = t.ppf(
            q=1 - alpha / 2,
            df=n - (p + 1))

        alpha_m = np.linalg.inv(np.dot(np.transpose(X_m), X_m))[p, p]

        hatE = Y - np.dot(X_m, Theta_m)

        NormaE = np.dot(np.transpose(hatE), hatE)
        CentralStatistics = Theta_m[p] / (alpha_m * NormaE) ** 0.5 * (n - (p + 1)) ** 0.5
        return Theta_m, CentralStatistics, quant_m

    p = 1
    while True:
        Theta, Z, quant = get_stat(p)
        if abs(Z) > quant:
            p += 1
        else:
            return *get_stat(p-1)[:2], p-1


def task_2(p):
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y = np.dot(X, theta) + EPS
    X_m = X[:40, :p + 1]
    Theta_m = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_m), X_m)), np.transpose(X_m)), Y)

    hatE = Y - np.dot(X_m, Theta_m)

    NormaE = np.dot(np.transpose(hatE), hatE)

    def get_interval(p, level):
        interval = []
        quant_m = t.ppf(
            q=1-(1-level)/2,
            df=n - (p + 1))
        for i in range(p + 1):
            left = Theta_m[i] - quant_m * (NormaE * np.linalg.inv(np.dot(np.transpose(X_m), X_m))[i, i]) ** 0.5 / (
                    n - (p + 1)) ** 0.5
            right = Theta_m[i] + quant_m * (NormaE * np.linalg.inv(np.dot(np.transpose(X_m), X_m))[i, i]) ** 0.5 / (
                    n - (p + 1)) ** 0.5

            interval.append([left, right])

        return np.array(interval)

    return get_interval(p, level=0.95), get_interval(p, level=0.99)


def task_3(p):
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y = np.dot(X, theta) + EPS
    hatY = np.dot(X, theta)
    X_m = X[:40, :p + 1]

    Theta_m = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_m), X_m)), np.transpose(X_m)), Y)

    def alpha(i):
        return np.dot(np.dot(X_m[i], np.linalg.inv(np.dot(np.transpose(X_m), X_m))), np.transpose(X_m[i]))

    hatE = Y - np.dot(X_m, Theta_m)

    NormaE = np.dot(np.transpose(hatE), hatE)

    def get_interval(p, level):
        left = np.zeros(n)
        right = np.zeros(n)
        quant_m = t.ppf(
            q=1 - (1 - level) / 2,
            df=n - (p + 1))
        for i in range(n):
            left[i] = hatY[i] - quant_m * (NormaE * alpha(i)) ** 0.5 / (n - (p + 1)) ** 0.5
            right[i] = hatY[i] + quant_m * (NormaE * alpha(i)) ** 0.5 / (n - (p + 1)) ** 0.5

        return np.array([left, right]).transpose()

    return get_interval(p, level=0.95), get_interval(p, level=0.99)


# print("-------------------1 номер-------------------")
# Theta, Z, p = task_1()
# print(f"Вектор-строку коэффициентов {Theta}\nМодель имеет порядок {p}\nZ = {Z**2}")
# print("---------------------------------------------")
#
# print("-------------------2 номер-------------------")
# level_095, level_099 = task_2(p)
# print(f"Для уровня надежности = 0.95\n{level_095}\n")
# print(f"Для уровня надежности = 0.99\n{level_099}")
# print("---------------------------------------------")
print(*task_3(2), sep="\n")