from .data_processor import GachaDataProcessor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import copy


class EquationProcessor:
    def __init__(self, uid, rank_type):
        self.uid = uid
        self.rank_type = rank_type
        self.data = GachaDataProcessor(uid, rank_type)

        df = pd.DataFrame(self.data.cleaned_data)
        self.equation_sort = df["gacha_type"].unique()
        self.equation_data = self.__get_equation()

    def modify_sort_equation(self, sort_index: list = [0, 1, 2, 3]):
        gacha_type = []
        if len(sort_index) != len(self.equation_sort):
            raise ValueError("Len sort index must be same as total equation")
        for i in sort_index:
            gacha_type.append({"gacha_type": self.equation_sort[i]})
        df = pd.DataFrame(gacha_type)
        self.equation_sort = df["gacha_type"].unique()
        self.equation_data = self.__get_equation()

    def prep_equation(self, raw):
        equation_data = copy.deepcopy(raw)
        A = []
        b = []
        for eq in equation_data:
            eq_dict = dict(eq)
            if "intercept" not in eq_dict:
                raise ValueError("Each equation must have an 'intercept'.")
            intercept = eq_dict.pop("intercept")
            b.append(intercept)
            A.append([eq_dict[var] for var in eq_dict])
        return np.array(A), np.array(b)

    def __check_diag_dominant(self, X, k):
        diag = np.abs(np.diag(X, k))
        non_diag = np.sum(np.abs(X), axis=1) - diag
        return np.all(diag > non_diag)

    def __get_equation(self):
        df = pd.DataFrame(self.data.cleaned_data)
        gacha_types = self.equation_sort
        rank_weight = {3: 1 / 90, 4: 1 / 9, 5: 1}
        equations = []
        k = 0
        min_len = 0

        for gct in gacha_types:
            subset = df[df["gacha_type"] == gct]
            if min_len == 0:
                min_len = len(subset)
            elif min_len != 0 and len(subset) < min_len:
                min_len = len(subset)

        for gct in gacha_types:
            df["rank_weight"] = df["rank_type"].map(rank_weight)
            subset = df[df["gacha_type"] == gct]
            subset = subset.iloc[:min_len]

            col_x = ["total_warp", "total_r5", "total_r4"]
            col_y = "month"

            X = subset[col_x]
            y = subset[col_y]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # model = Ridge(alpha=1.0)
            # model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            coefficients = model.coef_
            intercept = model.intercept_
            equation = f"{col_y} = {intercept:.2f} + " + " + ".join(
                [
                    f"({coeff:.2f} * {feature})"
                    for coeff, feature in zip(coefficients, X.columns)
                ]
            )

            equation_data = [("intercept", intercept)] + [
                (feature, coeff) for coeff, feature in zip(coefficients, X.columns)
            ]

            is_diag_dominant = self.__check_diag_dominant(model.coef_.reshape(1, -1), k)

            res = {
                "gacha type": gct,
                "equation data": equation_data,
                "equation string": equation,
                "MAE": mae,
                "R-squared": r2,
                "Diagonal Dominant": is_diag_dominant,
            }
            equations.append(res)
            k += 1
        return equations
