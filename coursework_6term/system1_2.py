from functools import partial as prt
from common import K as K_f
from common import D as D_f
from common import Element as E
import numpy as np

AlCl = 0
AlCl2 = 1
AlCl3 = 2
HCl = 3
H2 = 4

def calc_parameters(i, T):
    i = (i - 1) * 3
    K = [0, K_f[i + 1](T), K_f[i + 2](T), K_f[i + 3](T)]
    D = [
        D_f(E.AlCl, T),
        D_f(E.AlCl2, T),
        D_f(E.AlCl3, T),
        D_f(E.HCl, T),
        D_f(E.H2, T)
    ]
    P_g = [0, 0, 0, 10000, 0]
    return K, D, P_g

def F(K, D, Pg):
    return prt(_F, K, D, Pg)

def J(K, D, Pg):
    return prt(_J, K, D, Pg)

def prepare_system(i, T):
    (K, D, P_g) = calc_parameters(i, T)
    f = F(K, D, P_g)
    j = J(K, D, P_g)
    return f, j, (K, D, P_g)

def _F(K, D, Pg, Pe):
    return np.array([\
        (Pe[HCl] ** 2) - K[1] * (Pe[AlCl] ** 2) * Pe[H2],\
        (Pe[HCl] ** 2) - K[2] * Pe[AlCl2] * Pe[H2],\
        (Pe[HCl] ** 6) - K[3] * (Pe[AlCl3] ** 2) * (Pe[H2] ** 3),\
        D[HCl] * (Pg[HCl] - Pe[HCl]) + 2 * D[H2] * (Pg[H2] - Pe[H2]),\
        D[AlCl] * (Pg[AlCl] - Pe[AlCl]) + 2 * D[AlCl2] * (Pg[AlCl2] - Pe[AlCl2]) + 3 * D[AlCl3] * (Pg[AlCl3] - Pe[AlCl3]) + D[HCl] * (Pg[HCl] - Pe[HCl])\
    ])

def _J(K, D, Pg, Pe):
    return np.array([\
            [\
                0 - K[1] * 2 * Pe[AlCl] * Pe[H2],
                0,
                0,
                2 * Pe[HCl],
                0 - K[1] * (Pe[AlCl] ** 2)
            ],\
            [\
                0,\
                0 - K[2] * Pe[H2],\
                0,\
                2 * Pe[HCl],\
                0 - K[2] * Pe[AlCl2]\
            ],\
            [\
                0,\
                0,\
                0 - K[3] * 2 * Pe[AlCl3] * (Pe[H2] ** 3),\
                6 * (Pe[HCl] ** 5),\
                0 - K[3] * (Pe[AlCl3] ** 2) * 3 * (Pe[H2] ** 2)\
            ],\
            [\
                0,\
                0,\
                0,\
                0 - D[HCl],\
                0 - 2 * D[H2]\
            ],\
            [\
                0 - D[AlCl],\
                0 - 2 * D[AlCl2],\
                0 - 3 * D[AlCl3],\
                0 - D[HCl],\
                0\
            ]
        ])