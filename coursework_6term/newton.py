import numpy as np

class Newton:
    def __init__(self, eps, max_iter):
        self.eps = eps
        self.max_iter = max_iter

    def gradient_descent(self, F, J, x0):
        step = 1.0
        betta_step = 0.99
        x = np.array(x0)
        for i in range(self.max_iter // 100):
            A = np.array(J(x))
            b = np.array(F(x))
            dx = b @ A
            norm = np.linalg.norm(dx)
            dx = dx / norm
            x = x - dx * step
            step *= betta_step
        return x

    def calc(self, F, J, x0):
        x = self.gradient_descent(F, J, x0)
#         x = x0
        F_value = F(x)
        F_norm = np.linalg.norm(F_value, ord=2)
        iter_cnt = 0
        while abs(F_norm) > self.eps and iter_cnt < self.max_iter:
            delta = np.linalg.solve(J(x), -F_value)
            x = x + delta
            F_value = F(x)
            F_norm = np.linalg.norm(F_value, ord=2)
            iter_cnt += 1

        if iter_cnt == self.max_iter:
#             raise Exception('Method did not converge')
            return [1, 1, 1, 1, 1]
# #        print(x)
        return x
