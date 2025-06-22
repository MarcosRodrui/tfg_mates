from kan import KAN
from torch import device, nn

from edp_lib.edp_solver import EDPSolver

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        # Se configura el MLP utilizando la configuración
        super(MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
       
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class EDPSolverMlp(EDPSolver):
    """
    Esta clase soluciona EDPs utilizando un MLP
    """

    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.hidden_dim = 64
        self.output_dim = 1
        self.num_hidden_layers = 2

    def configure_mlp_settings(self, input_dim=None, hidden_dim=None, output_dim=None, num_hidden_layers=None):
        """
        Configura los parámetros del MLP
        """
        if input_dim is not None:
            self.input_dim = input_dim
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        if output_dim is not None:
            self.output_dim = output_dim
        if num_hidden_layers is not None:
            self.num_hidden_layers = num_hidden_layers

    def get_model(self) -> nn.Module:
        """
        Devuelve el modelo del MLP
        """
        return MLP(self.input_dim, self.hidden_dim, self.output_dim, self.num_hidden_layers).to(device=self.device)
