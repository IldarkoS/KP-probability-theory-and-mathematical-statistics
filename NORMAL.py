import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, chi2

EPS = np.array(
    [0.5825356586711,
     0.779136241919401,
     0.069108921157575,
     0.133724868503062,
     -0.185744449477847,
     -1.1777860134251,
     0.739550630925385,
     0.885757877426329,
     0.244582137320006,
     0.791348132291261,
     0.542194571144387,
     0.140962632813154,
     0.914079982466143,
     1.11493101721298,
     -1.19059423052731,
     0.170760986395866,
     -0.09412406540652,
     1.07532818795774,
     0.93684563929557,
     -3.13208181730096,
     0.199991156127484,
     -0.917857248143994,
     -0.537772490298977,
     0.153056138636936,
     -0.0471328256533068,
     -0.453000226511304,
     0.522747089851915,
     -1.71017005474495,
     -1.88558100989699,
     -0.769550314820384,
     0.519057251415655,
     0.182434800723214,
     -0.175139291126682,
     1.84149902650684,
     2.1151541477258,
     0.0187015347546949,
     0.398284526834759,
     1.35459474080133,
     1.34173415315512,
     -0.587816367648232
     ])

n = 40
sigma = 1.6 ** 0.5
theta0 = 10
theta1 = -2
theta2 = 1
theta3 = -0.07

theta = np.array([theta0, theta1, theta2, theta3])
# theta = np.array([20, -2, -1, -0.06])
# theta = np.array([11, -3, 6, 0.12])


def y(th0, th1, th2, th3, t):
    return th3 * t ** 3 + th2 * t ** 2 + th1 * t + th0


def task_1():
    X = np.zeros((n, 6))

    for i in range(n):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3
        X[i][4] = (-4 + (i + 1) * 8 / n) ** 4
        X[i][5] = (-4 + (i + 1) * 8 / n) ** 5

    Y = np.dot(X[:n, :4], theta) + EPS

    def get_stat(p, alpha=0.05):
        X_m = X[:n, :p + 1]

        Theta_m = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_m), X_m)), np.transpose(X_m)), Y)

        quant_m = t.ppf(
            q=1 - alpha / 2,
            df=n - (p + 1))

        alpha_m = np.linalg.inv(np.dot(np.transpose(X_m), X_m))[p, p]

        hatE = Y - np.dot(X_m, Theta_m)

        NormaE = np.dot(np.transpose(hatE), hatE)
        CentralStatistics = Theta_m[p] / (alpha_m * NormaE) ** 0.5 * (n - (p + 1)) ** 0.5
        # print(p, Theta_m, CentralStatistics, quant_m)
        return Theta_m, CentralStatistics, quant_m

    p = 2
    while True:
        Theta, Z, quant = get_stat(p)
        if abs(Z) > quant:
            p += 1
        else:
            return *get_stat(p - 1)[:2], p - 1


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
            q=1 - (1 - level) / 2,
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


def task_4(theta_m, level_095, level_099):
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y_without_noise = np.dot(X, theta)
    Y_with_noise = np.dot(X, theta) + EPS

    fig = plt.figure(figsize=[14, 10])
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x := X.transpose()[1], Y_without_noise, label="Истинный полезный сигнал")
    ax.plot(x, Y_with_noise, label="Набор наблюдений", ls='None', marker='.')
    y_check = np.zeros(n)
    for i in range(n):
        for j in range(len(theta_m)):
            y_check[i] += theta_m[j] * (x[i] ** j)
    ax.plot(x, y_check, label="Оценка полезного сигнала")
    ax.plot(x, [level_095[i][0] for i in range(n)], label="Доверительный интервал нижний для alpha = 0.95")
    ax.plot(x, [level_095[i][1] for i in range(n)], label="Доверительный интервал верхний для alpha = 0.95")
    ax.plot(x, [level_099[i][0] for i in range(n)], label="Доверительный интервал нижний для alpha = 0.99")
    ax.plot(x, [level_099[i][1] for i in range(n)], label="Доверительный интервал верхний для alpha = 0.99")

    ax.legend()


def task_5():
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y = np.dot(X, theta) + EPS
    X_m = X[:40, :p + 1]
    Theta_m = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_m), X_m)), np.transpose(X_m)), Y)
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(1, 1, 1)
    # Остатки регрессии
    hatE = Y - np.dot(X_m, Theta_m)
    a = ax.hist(hatE, bins=5, density=True, label="Гистограмма остатков регрессии")
    ax.legend()
    return a


def task_6(p):
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

    dispersion = NormaE / (n - (p + 1))

    return dispersion


def task_7(gistogramma):
    # print(gistogramma)
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

    sigma = NormaE / n

    def T(gistogramma):
        T = 0
        for i in range(len(gistogramma[0]) - 1):
            T += (pk := (norm.cdf(gistogramma[1][i + 1] / sigma ** 0.5) - norm.cdf(gistogramma[1][i] / sigma ** 0.5)) -
                        gistogramma[0][i]) ** 2 / pk
        return T * n

    TZn = T(gistogramma)
    return "Распределение ошибок нормальное" if 0 < TZn < chi2.ppf(0.95,
                                                                   5) else "Распределение ошибок не является нормальным"


print("-------------------1 номер-------------------")
Theta, Z, p = task_1()
print(f"Вектор-строка коэффициентов {Theta}\nМодель имеет порядок {p}\nZ = {Z}")
print("---------------------------------------------")
#
print("-------------------2 номер-------------------")
level_095, level_099 = task_2(p)
print(f"Для уровня надежности = 0.95\n{level_095}\n")
print(f"Для уровня надежности = 0.99\n{level_099}")
print("---------------------------------------------")
print("-------------------3 номер-------------------")
level_095_for_signal, level_099_for_signal = task_3(p)
print(f"Для уровня надежности = 0.95\n{level_095_for_signal}\n")
print(f"Для уровня надежности = 0.99\n{level_099_for_signal}")
print("---------------------------------------------")
print("-------------------4 номер-------------------")
task_4(Theta, level_095_for_signal, level_099_for_signal)
print("---------------------------------------------")
print("-------------------5 номер-------------------")
gistogramma = task_5()
print("---------------------------------------------")
print("-------------------6 номер-------------------")
dispersion = task_6(p)
print(f"Несмещенная норма дисперсии: {dispersion}")
print("---------------------------------------------")
print("-------------------7 номер-------------------")
print(task_7(gistogramma))
print("---------------------------------------------")
plt.show()
