from setup.data_processor import GachaDataProcessor
from sympy import symbols, Eq


class EquationGaussSeidel:

    def __init__(self, data_processor: GachaDataProcessor):
        self.data = data_processor
