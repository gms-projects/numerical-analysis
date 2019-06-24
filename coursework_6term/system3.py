from functools import partial as prt
from common import K as K_f
from common import D as D_f
from common import Element as E
import numpy as np

T = 1373.15

AlCl3 = 0
GaCl = 1
NH3 = 2
HCl = 3
H2 = 4
x = 5
N2 = 6

def calc_parameters(x_g, i):
    K = [0, 1, 2, 3, 4, 5, 6, 7, 8, K_f[9](T), K_f[10](T)]
    D = [
        D_f(E.AlCl3, T),
        D_f(E.GaCl, T),
        D_f(E.NH3, T),
        D_f(E.HCl, T),
        D_f(E.H2, T),
    ]
    Pg = [30 * x_g, 30 * (1 - x_g), 1500, 0, 9847 * i]
    return K, D, Pg

def prepare_system(x_g, i):
    (K, D, Pg) = calc_parameters(x_g, i)
    f = F(K, D, Pg)
    j = J(K, D, Pg)
    return f, j, (K, D, Pg)

def F(K, D, Pg):
    return prt(_F, K, D, Pg)

def J(K, D, Pg):
    return prt(_J, K, D, Pg)

def _F(K, D, Pg, Pe):
    x = Pe[5]
    return np.array([\
            Pe[AlCl3] * Pe[NH3] - K[9] * x * Pe[HCl] ** 3,\
            Pe[GaCl] * Pe[NH3] - K[10] * (1 - x) * Pe[HCl] * Pe[H2],\
            D[HCl] * (Pg[HCl] - Pe[HCl]) + 2 * D[H2] * (Pg[H2] - Pe[H2]) + 3 * D[NH3] * (Pg[NH3] - Pe[NH3]),\
            3 * D[AlCl3] * (Pg[AlCl3] - Pe[AlCl3]) + D[GaCl] * (Pg[GaCl] - Pe[GaCl]) + D[HCl] * (Pg[HCl] - Pe[HCl]),\
            D[AlCl3] * (Pg[AlCl3] - Pe[AlCl3]) + D[GaCl] * (Pg[GaCl] - Pe[GaCl]) - D[NH3] * (Pg[NH3] - Pe[NH3]),\
            (1 - x) * D[AlCl3] * (Pg[AlCl3] - Pe[AlCl3]) - x * D[GaCl] * (Pg[GaCl] - Pe[GaCl])\
        ])

def _J(K, D, Pg, Pe):
    x = Pe[5]
    return np.array([\
            [\
             Pe[NH3],\
             0,\
             Pe[AlCl3],\
             0 - 3 * K[9] * x * Pe[HCl] ** 2,\
             0,\
             0 - K[9] * Pe[HCl] ** 3\
            ],\
            [\
             0,\
             Pe[NH3],\
             Pe[GaCl],\
             0 - K[10] * (1 - x) * Pe[H2],\
             0 - K[10] * (1 - x) * Pe[HCl],\
             K[10] * Pe[HCl] * Pe[H2]\
            ],\
            [\
             0,\
             0,\
             0 - 3 * D[NH3],\
             0 - D[HCl],\
             0 - 2 * D[H2],\
             0\
            ],\
            [\
             0 - 3 * D[AlCl3],\
             0 - D[GaCl],\
             0,\
             0 - D[HCl],\
             0,\
             0\
            ],\
            [\
             0 - D[AlCl3],\
             0 - D[GaCl],\
             D[NH3],\
             0,\
             0,\
             0\
            ],\
            [\
             x * D[AlCl3] - D[AlCl3],\
             x * D[GaCl],\
             0,\
             0,\
             0,\
             - D[AlCl3] * (Pg[AlCl3] - Pe[AlCl3]) - D[GaCl] * (Pg[GaCl] - Pe[GaCl])\
            ]
        ]
    )