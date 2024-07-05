"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
from typing import Any, Dict, List

import numpy as np
from sympy import symbols, Symbol
from sympy.core.expr import Expr

import geoCosiCorr3D.geoCore.constants as C


class FuncModel:
    def __init__(self, model_name: str = C.RsmRefinementModels.LINEAR) -> None:
        self.model_name = model_name
        if self.model_name == C.RsmRefinementModels.LINEAR:
            self.model = self.multivariate_linear_poly()
        elif self.model_name == C.RsmRefinementModels.QUADRATIC:
            self.model = self.multivariate_quadratic_poly()
        else:
            raise ValueError(f"Model {self.model_name} is not supported")

    def multivariate_linear_poly(self) -> Expr:
        x, y, a, b, c = symbols('x y a b c')
        return a * x + b * y + c

    def multivariate_quadratic_poly(self) -> Expr:
        x, y, a, b, c, d, e, f = symbols('x y a b c d e f')
        return a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f

    def evaluate_model(self, values: Dict[Symbol, Any]) -> Any:
        return self.model.subs(values)


def cost_func(observations: List[C.Observation], dim: int, func_model: FuncModel, params: List[float]):
    v = np.zeros(len(observations))
    for i, obs in enumerate(observations):
        if func_model.model_name == 'linear':
            f = func_model.evaluate_model(
                {symbols('a'): params[0], symbols('b'): params[1], symbols('c'): params[2],
                 symbols('x'): obs.COL, symbols('y'): obs.LIN})
        elif func_model.model_name == 'quadratic':
            f = func_model.evaluate_model(
                {symbols('a'): params[0], symbols('b'): params[1], symbols('c'): params[2],
                 symbols('d'): params[3], symbols('e'): params[4], symbols('f'): params[5],
                 symbols('x'): obs.COL, symbols('y'): obs.LIN})
        else:
            raise ValueError(f"Model {func_model.model_name} is not supported")
        v[i] = obs.WEIGHT * (obs.DU[dim] - f)
    return v


def wlsq_gauss_markov(x, y, w, error):
    w_square = w ** 2
    alpha1 = np.sum(w_square * x ** 2)
    alpha2 = np.sum(w_square * y * x)
    alpha3 = np.sum(w_square * x)

    beta_2 = np.sum(w_square * y ** 2)
    beta_3 = np.sum(w_square * y)

    gamma_3 = np.sum(w_square)

    delta1 = np.sum(w_square * x * error)
    delta2 = np.sum(w_square * y * error)
    delta3 = np.sum(w_square * error)

    mat = np.array([[alpha1, alpha2, alpha3],
                    [alpha2, beta_2, beta_3],
                    [alpha3, beta_3, gamma_3]])

    delta = np.array([delta1, delta2, delta3]).T
    P = np.dot(np.linalg.inv(mat), delta)

    return P
