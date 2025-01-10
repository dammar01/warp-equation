import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EquationGaussSeidel:
    def calc(self, A, b, tol=1e-10, max_iter=100):
        n = len(A)
        x = np.zeros_like(b, dtype=float)
        if np.any(np.diag(A) == 0):
            raise ValueError(
                "Matrix A contains zero on its diagonal, cannot apply Gauss-Seidel."
            )
        iteration_data = []

        for iteration in range(1, max_iter + 1):
            x_new = np.copy(x)
            for i in range(n):
                sum_ = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_) / A[i, i]

            # Calculate error
            error = np.linalg.norm(x_new - x, ord=np.inf)
            iteration_info = {
                "iteration": iteration,
                "x_values": x_new.copy(),
                "error": error,
            }

            # Check if converged
            if error < tol:
                iteration_info["status"] = "Converged"
                iteration_data.append(iteration_info)
                return pd.DataFrame(iteration_data)

            # Otherwise, continue
            iteration_info["status"] = "Ongoing"
            iteration_data.append(iteration_info)
            x = x_new

        # If we exit the loop without convergence
        iteration_data[-1]["status"] = "Not Converged"
        return pd.DataFrame(iteration_data)

    def visualize(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(df["iteration"], df["error"], marker="o", color="b", label="Error")
        plt.axhline(y=0, color="r", linestyle="--", label="Konvergen (error=0)")

        plt.title("Error visualization in all iteration using Gauss-Seidel")
        plt.xlabel("Iterasi")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        plt.show()
