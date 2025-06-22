from kan import KAN
from torch import nn

from edp_lib.edp_solver import EDPSolver

class EDPSolverKan(EDPSolver):
    """
    Esta clase soluciona EDPs utilizando una KAN
    """

    def __init__(self):
        super().__init__()

        self.grid = 5
        self.k_spline = 3
        self.kan_width = [2, 2, 1]

    def configure_kan_settings(self, grid=None, k_spline=None, kan_width=None):
        """
        Se configuran los parámetros que son únicos de la KAN
        """
        if grid is not None:
            self.grid = grid
        if k_spline is not None:
            self.k_spline = k_spline
        if kan_width is not None:
            if self.check_valid_width(kan_width):
                self.kan_width = kan_width

    def check_valid_width(self, kan_width: list[int]) -> bool:
        """
        Se comprueba que la configuración de la anchura es válida
        """
        if len(kan_width) < 2:
            print(f"Width of the kan network should at least be 2.")
            return False

        if kan_width[0] != len(self.variables):
            print(f"First layer of the KAN should be the same as the number of variables.")
            return False

        if kan_width[-1] != len(self.outputs):
            print(f"Last layer of the KAN should be the same as the number of variables.")
            return False

        return True

    def save_model(self, model_file):
        """
        Se guarda el modelo de la KAN en el fichero indicado
        """
        self.model.saveckpt(model_file)

    def get_model(self) -> nn.Module:
        """
        Se devuelve el modelo KAN
        """
        return KAN(width=self.kan_width, grid=5, k=3, device=self.device, auto_save=False)

    def periodic_update(self):
        """
        Se actualiza el grid utilizando la entrada (necesario para actualizar
        las nudos de las splines)
        """
        self.model.update_grid_from_samples(self.var_i)