import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, chi2

EPS = np.array(
    [-1.57570224093566,
     0.026817652058368,
     -0.327083400768208,
     0.384457963489581,
     1.13092869940426,
     0.496851264054229,
     0.907339535512404,
     0.100098748083528,
     2.49180275180069,
     -0.0728842702874212,
     0.826877669546407,
     3.12941087626926,
     -0.290112040544173,
     0.273222919888945,
     0.945511583398219,
     -1.76916070866538,
     1.40708789619999,
     -2.48341591842828,
     -2.77630227137947,
     3.13991491474395,
     1.10107674176732,
     -0.257722179163009,
     -1.73620920255182,
     -1.84146124965671,
     -1.43220867624349,
     -0.657551650718961,
     0.186855475002093,
     -2.35604061658699,
     1.03240424046648,
     0.334456144504168,
     0.470946667999632,
     -1.76644417119997,
     1.23171894652158,
     -2.41344098737007,
     -0.553993401124847,
     0.712286159261229,
     -0.229382238056625,
     2.02815685186965,
     2.95489535309773,
     0.147225408140916
     ])

n = 40
sigma = 1.6 ** 0.5
theta0 = 10
theta1 = -2
theta2 = 1
theta3 = -0.07

theta = np.array([theta0, theta1, theta2, theta3])


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
    print(Y)
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

    p = 1
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


def task_3(p, hatTheta):
    X = np.zeros((40, 4))

    for i in range(40):
        X[i][0] = 1
        X[i][1] = (-4 + (i + 1) * 8 / n)
        X[i][2] = (-4 + (i + 1) * 8 / n) ** 2
        X[i][3] = (-4 + (i + 1) * 8 / n) ** 3

    Y = np.dot(X, theta) + EPS
    X_m = X[:40, :p + 1]

    hatY = np.dot(X_m, hatTheta)

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

    X_m = X[:40, :len(theta_m)]

    Y_without_noise = np.dot(X, theta)
    Y_with_noise = np.dot(X, theta) + EPS

    fig = plt.figure(figsize=[14, 10])
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x := X.transpose()[1], Y_without_noise, label="Истинный полезный сигнал")
    ax.plot(x, Y_with_noise, label="Набор наблюдений", ls='None', marker='.')
    y_check = np.dot(X_m, theta_m)
    ax.plot(x, y_check, label="Оценка полезного сигнала", ls='None', marker='.')
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

    # print(NormaE)
    # print(hatE, np.var(hatE))
    # dispersion = np.var(hatE)
    dispersion = NormaE/n

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
    # print(sigma)
    def T(gistogramma):
        T = 0
        for i in range(len(gistogramma[0]) - 1):
            T += (pk := (norm.cdf(gistogramma[1][i + 1] / sigma ** 0.5) - norm.cdf(gistogramma[1][i] / sigma ** 0.5)) -
                        gistogramma[0][i]) ** 2 / pk
        return T * n
    TZn = T(gistogramma)
    # print(TZn)
    # print(chi2.ppf(0.95, 5))
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
level_095_for_signal, level_099_for_signal = task_3(p, Theta)
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
print(f"Оценка максимального правдоподобия дисперсии: {dispersion}")
print("---------------------------------------------")
print("-------------------7 номер-------------------")
print(task_7(gistogramma))
print("---------------------------------------------")
plt.show()
