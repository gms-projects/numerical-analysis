import math
from enum import Enum
import pandas as pd
import csv
import re


P_a = 10 ** 5
R = 8.314

class Element(Enum):
    AlCl  = 0
    AlCl2 = 1
    AlCl3 = 2
    GaCl  = 3
    GaCl2 = 4
    GaCl3 = 5
    NH3   = 6
    H2    = 7
    HCl   = 8
    N2    = 9
    Al    = 10
    Ga    = 11
    AlN   = 12
    GaN   = 13


# Расчет энергии Гиббса компоненты
def G(el, T):
    x = T / 10 ** 4

    # аппроксимация
    (f1, f2, f3, f4, f5, f6, f7) = readRow(el, ["f1", "f2", "f3", "f4", "f5", "f6", "f7"])
    H = get(el, "H(298)")

    F = f1 + f2 * math.log(x) + f3 / (x ** 2) + f4 / x + f5 * x + f6 * x ** 2 + f7 * x ** 3
    return H - F * T

# Расчет коэффициентов диффузии газообразных компонент
def D(el, T):
    # сечение столкновения молекул
    (sigma_N2, sigma_el) = readCol("sigma", [Element.N2, el])
    sigma = (sigma_N2 + sigma_el) / 2

    # глубина потенциальной ямы энергии взаимодействия
    (epsil_N2, epsil_el) = readCol("epsil", [Element.N2, el])
    epsil = (epsil_N2 * epsil_el) ** 0.5

    # средняя молярная масса
    (mu_N2, mu_el) = readCol("mu", [Element.N2, el])
    mu = 2 * (mu_N2 * mu_el) / (mu_N2 + mu_el)

    # аппроксимация интеграла столкновений
    theta = 1.074 * ((T / epsil) ** (-0.1604))
    return 0.02628 * (T ** 1.5) / (P_a * sigma * theta * (mu ** 0.5))


# Константы равновесия
def ex(d, T):
    return math.exp(-d / (R * T))

def K1(T):
    deltaG1 = 2 * G(Element.Al, T) + 2 * G(Element.HCl, T) - 2 * G(Element.AlCl, T) - G(Element.H2, T)
    return ex(deltaG1, T) / P_a

def K2(T):
    deltaG2 = G(Element.Al, T) + 2 * G(Element.HCl, T) - G(Element.AlCl2, T) - G(Element.H2, T)
    return ex(deltaG2, T)

def K3(T):
    deltaG3 = 2 * G(Element.Al, T) + 6 * G(Element.HCl, T) - 2 * G(Element.AlCl3, T) - 3 * G(Element.H2, T)
    return ex(deltaG3, T) * P_a

def K4(T):
    deltaG4 = 2 * G(Element.Ga, T) + 2 * G(Element.HCl, T) - 2 * G(Element.GaCl, T) - G(Element.H2, T)
    return ex(deltaG4, T) / P_a

def K5(T):
    deltaG5 = G(Element.Ga, T) + 2 * G(Element.HCl, T) - G(Element.GaCl2, T) - G(Element.H2, T)
    return ex(deltaG5, T)

def K6(T):
    deltaG6 = 2 * G(Element.Ga, T) + 6 * G(Element.HCl, T) - 2 * G(Element.GaCl3, T) - 3 * G(Element.H2, T)
    return ex(deltaG6, T) * P_a

def K9(T):
    deltaG9 = G(Element.AlCl3, T) + G(Element.NH3, T) - G(Element.AlN, T) - 3 * G(Element.HCl, T)
    return ex(deltaG9, T) / P_a

def K10(T):
    deltaG10 = G(Element.GaCl, T) + G(Element.NH3, T) - G(Element.GaN, T) - G(Element.HCl, T) - G(Element.H2, T)
    return ex(deltaG10, T)

K = [None, K1, K2, K3, K4, K5, K6, None, None, K9, K10]

# Преобразование .dat в .csv
with open('Bank_TD_Fragment.dat') as dat_file, open('Bank.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for line in dat_file:
        csv_writer.writerow(re.split('\s+', line))

# Считывание .csv
data = pd.read_csv('Bank.csv')

# Получение данных из .dat файла
def get(el, col):
    return data[col][el.value]

def readRow(el, cols):
    return tuple([get(el, col) for col in cols])

def readCol(col, elems):
    return tuple([get(el, col) for el in elems])
