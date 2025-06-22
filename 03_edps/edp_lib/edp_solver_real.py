from torch import nn
import numpy as np

from edp_lib.edp_solver import EDPSolver

class EDPSolverReal(EDPSolver):
    """
    Esta clase abstrea el simular la solución real de una EDP
    """
    def __init__(self):
        super().__init__()

        self.real_solution = dict()

    def get_model(self) -> nn.Module:
        """
        Esta función no se debería llamar nunca porque se usa en el entrenamiento
        y esta clase no se entrena.
        """
        raise Exception("This class does not need to be trained.")

    def set_real_solution(self, real_solution: dict):
        """
        Configura la solución real
        """
        self.real_solution = real_solution

    def get_solution(self):
        """
        Devuelve la solución que se corresponde con la solución real
        """
        if len(self.real_solution) == 0:
            print("Solution cannot be generated, the real solution has not been provided.")
            return ()

        solutions = [None] * len(self.outputs)
        for output, idx in self.outputs.items():
            def sol(*args, real_sol=self.real_solution):
                grids = np.meshgrid(*args, indexing="ij")
                return real_sol[output](*grids)

            solutions[idx] = sol

        return tuple(solutions)
