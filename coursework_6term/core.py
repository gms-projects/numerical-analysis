import math
import numpy as np
from functools import partial as prt
from common import R
from newton import Newton
import system1_2
import system3


class Task:
    def __init__(self, solve_system, x0):
        self.solve_system = solve_system
        self.x0 = x0

    def calc_interfacialG(self, D, Pg, Pe, T):
        G = (D * abs(Pg - Pe)) / (1000 * R * T * 0.01)
        return G

    def interfacial_G(self, T):
        (D, Pg, Pe) = self.solve_system(T)
        return [
            self.calc_interfacialG(D[i], Pg[i], Pe[i], T)
            for i in range(0, len(D))
        ]


class Task1(Task):
    def __init__(self, x0):
        super().__init__(self.solve_system, x0)
        self.newton = Newton(1e-9, 10**3)

    def calc_V(self, T):
        G = self.interfacial_G(T)
        y = sum(G[:3])
        x = y * (26.981539 / 2690) * 10**9
        return x

    def solve_system(self, T):
        (F, J, (K, D, Pg)) = system1_2.prepare_system(1, T)
        Pe = self.newton.calc(F, J, self.x0)
        self.x0 = Pe
        return (D, Pg, Pe)


class Task2(Task):
    def __init__(self, x0):
        super().__init__(self.solve_system, x0)
        self.newton = Newton(1e-8, 10**4)

    def calc_V(self, T):
        G = self.interfacial_G(T)
        y = sum(G[:3])
        x = y * (69.723 / 5900) * 10**9
        return x

    def solve_system(self, T):
        (F, J, (K, D, Pg)) = system1_2.prepare_system(2, T)
        Pe = self.newton.calc(F, J, self.x0)
        self.x0 = Pe
        return (D, Pg, Pe)


class Task3(Task):
    def __init__(self, x0):
        super().__init__(self.solve_system, x0)
        self.newton = Newton(1e-9, 10**3)


    def interfacial_G(self, x_g, i):
        (D, Pg, Pe) = self.solve_system(x_g, i)
        return [
            self.calc_interfacialG(D[i], Pg[i], Pe[i], 1373.15)
            for i in range(0, 5)
        ]

    def calc_V(self, x_g, i):
        G = self.interfacial_G(x_g, i)
        return (G[0] * (40.988 / 3200) + G[1] *
                          (83.730 / 6150)) * 10**9

    def solve_system(self, x_g, i):
        (F, J, (K, D, Pg)) = system3.prepare_system(x_g, i)
        Pe = self.newton.calc(F, J, self.x0)
        self.x0 = Pe
        return (D, Pg, Pe)

    def al_part(self, x_g, i):
        (D, Pg, Pe) = self.solve_system(x_g, i)
        return Pe[5]
