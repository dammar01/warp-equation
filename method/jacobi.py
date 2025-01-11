import numpy as np
import pandas as pd


class EquationJacobi:
    def calc(self, A, b, tol=1e-10, max_iter=100):
        n = len(b)
        x = np.zeros_like(b, dtype=float)
        iteration_data = []

        for iteration in range(1, max_iter + 1):
            x_new = np.copy(x)
            for i in range(n):
                sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_) / A[i, i]
            error = np.linalg.norm(x_new - x, ord=np.inf)
            iteration_info = {
                "iteration": iteration,
                "Total_warp": x_new[0],
                "Total_r5": x_new[1],
                "Total_r4": x_new[2],
                "error": error,
            }

            if error < tol:
                iteration_info["status"] = "Converged"
                iteration_data.append(iteration_info)
                return pd.DataFrame(iteration_data)

            iteration_info["status"] = "Ongoing"
            iteration_data.append(iteration_info)
            x = x_new

        iteration_data[-1]["status"] = "Not Converged"
        return pd.DataFrame(iteration_data)
