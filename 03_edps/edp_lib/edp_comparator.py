import os
from typing import Tuple
import numpy as np

from edp_lib.edp_solver import EDPSolver
from edp_lib.edp_solver_kan import EDPSolverKan
from edp_lib.edp_solver_mlp import EDPSolverMlp
from edp_lib.edp_solver_real import EDPSolverReal
from edp_lib.edp_checker import EDPChecker

class EDPComparator:
    """
    Esta clase se utiliza para comparar diferentes resolutores de EDPs
    utilizando el mismo método (PINN). Esta configurada para comparar
    KAN y MLP.
    """

    def __init__(self, name) -> None:
        self.name = name

        self.kan_solver = EDPSolverKan()
        self.mlp_solver = EDPSolverMlp()
        self.real_solver = EDPSolverReal()
        self.solvers = {
            "kan": self.kan_solver,
            "mlp": self.mlp_solver,
            "real": self.real_solver
        }
        self.trained = False

        self.variables = {}
        self.var_ranges = {}

        self.edp_checker = EDPChecker()

    def configure_variables(self, variables: dict[str, Tuple]):
        """
        Se configuran las variables de todos los modelos que
        se van a comparar.
        """
        for idx, v in enumerate(variables.keys()):
            self.variables[v] = idx
        self.var_ranges = variables

        for solver in self.solvers.values():
            solver.configure_variables(variables)
        self.edp_checker.configure_variables(variables)

    def configure_outputs(self, outputs: list[str]):
        """
        Se configuran las salidas de todos los modelos
        que se comparan.
        """
        for solver in self.solvers.values():
            solver.configure_outputs(outputs)
        self.edp_checker.configure_outputs(outputs)

    def configure_pde(self, pde_list: list) -> bool:
        """
        Se configura la EDP que va a ser resuelta por
        los modelos.
        """
        ret = True
        for solver in self.solvers.values():
            ret &= solver.configure_pde(pde_list)
        return ret

    def configure_boundaries(self, boundary_list: list) -> bool:
        """
        Se configuran las condiciones de frontera de la EDP
        que van a resolver los modelos
        """
        ret = True
        for solver in self.solvers.values():
            ret &= solver.configure_boundaries(boundary_list)
        return ret

    def configure_train_settings(self, solver_name, steps=None, alpha=None):
        """
        Se actualizan los hiperparámetros de todos los modelos
        """
        if solver_name not in self.solvers.keys():
            print("Povided name is not a solver.")
            return
        self.solvers[solver_name].configure_train_settings(steps, alpha)

    def configure_kan_settings(self, grid=None, k_spline=None, kan_width=None):
        """
        Se configuran los hiperparámetros de la KAN
        """
        self.kan_solver.configure_kan_settings(grid, k_spline, kan_width)

    def configure_mlp_settings(self, input_dim=None, hidden_dim=None, output_dim=None, num_hidden_layers=None):
        """
        Se configuran los hiperparámetros del MLP
        """
        self.mlp_solver.configure_mlp_settings(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def start_device(self):
        """
        Se inician los dispositivos de todos los modelos
        """
        for solver in self.solvers.values():
            solver.start_device()

    def train(self):
        """
        Se realiza el entrenamiento de la KAN y el MLP
        """
        self.kan_solver.train(f"saved_models/comparator/{self.name}_kan", f"imgs/{self.name}_kan_pde_loss.png")
        self.mlp_solver.train(f"saved_models/comparator/{self.name}_mlp", f"imgs/{self.name}_mlp_pde_loss.png")

    def set_real_solution(self, real_solution: dict):
        """
        Se configura la solución real
        """
        self.real_solver.set_real_solution(real_solution)

    def compare(self, output_folder, with_difference=True, force_train=False):
        """
        Se entrenan ambos modelos. Primero se realiza el entrenamiento en caso de que no se haya hecho.
        Luego, se generan las imagenes que contienen los resultados.
        """
        if not self.trained or force_train:
            self.start_device()
            self.train()
            self.trained = True

        path = os.path.join("graphs", self.name)
        if with_difference:
            self.edp_checker.create_graph(self.solvers, os.path.join(output_folder, path), "real")
        else:
            self.edp_checker.create_graph(self.solvers, os.path.join(output_folder, path))

    def get_results(self, force_train=False):
        """
        Devuelve una comparación entre los resultados generados
        """
        if not self.trained or force_train:
            self.start_device()
            self.train()
            self.trained = True

        self.edp_checker.get_results(self.solvers, "real")

    def get_funct_pred(self, data, solver="kan", force_train=False):
        """
        Devuelve la predicción por el modelo seleccionado
        """
        if not self.trained or force_train:
            self.start_device()
            self.train()
            self.trained = True

        if solver not in self.solvers.keys():
            print(f"{solver} does not exists.")
            return None

        solutions = self.solvers[solver].get_solution()
        pred = []
        for solution in solutions:
            if solution is None:
                continue
            pred.append(solution(*data))

        return pred