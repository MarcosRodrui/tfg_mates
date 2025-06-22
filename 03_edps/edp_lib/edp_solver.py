from abc import abstractmethod
from typing import Tuple
from kan import LBFGS
import inspect
import torch
from torch import autograd, nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class EDPSolver:
    """
    Clase principal de la que heredan REAL, KAN Y MLP. Resuelve una
    EDP utilizando el método de PINN
    """
    def __init__(self):
        # Este diccionario asocia una variable de la EDP con un identificador numérico
        self.variables = dict()
        # Este diccionario asocia una variable de la EDP con su rango de definición
        self.var_ranges = dict()
        # Este diccionario asocia una salida de la EDP con un identificador numérico
        self.outputs = dict()
        # Lista con las EDPs que marcan las condiciones del interior
        self.pde_list = list()
        # Lista con las condiciones de la frontera
        self.boundary_list = list()

        self.steps = 20
        self.alpha = 0.01
        self.device = ""
        self.interior_pts = 50

        self.max_order = 0
        self.trained = False

    def configure_variables(self, variables: dict[str, Tuple]):
        """
        Configura las variables de las EDPs y su rango de definición
        """
        for idx, v in enumerate(variables.keys()):
            self.variables[v] = idx

        self.var_ranges = variables

    def configure_outputs(self, outputs: list[str]):
        """
        Configura las salidas de la EDP. Es decir, cuando la EDP
        tiene varias funciones: u,v... Si solo tiene una función
        outputs será un array de un elemento [u]
        """
        for idx, o in enumerate(outputs):
            self.outputs[o] = idx

    def check_valid_pde_input(self, input: str) -> bool:
        """
        Comprueba si la string de entrada es una variable, una salida
        o una derivada parcial de alguna salida. Si es alguna de
        esas cosas devuelve True, sino devuelve False.
        """
        if input in self.variables:
            return True

        if input in self.outputs:
            return True

        # Cuenta el orden máximo de cualquiera de las derivadas
        # Esto es relevante porque se calcularán derivadas hasta este orden
        # en el entrenamiento
        self.max_order = 0

        # Se comprueba si el input pasado es una derivada
        for output in self.outputs:
            prefix = output + "_"
            if input.startswith(prefix):
                sufix = input[len(prefix):]
                count = 0
                for s in sufix:
                    count += 1
                    self.max_order = max(count, self.max_order)
                    if s not in self.variables.keys():
                        return False
                return count > 0

        return False

    def configure_pde(self, pde_list: list) -> bool:
        """
        Define la EDP igualada a 0. Por ejemplo:
        [u_t - u_x + 3*u_xx] Definiría una EDP sobre una función u(x,t),
        donde las derivadas parciales u_t, u_x, u_xx están implicadas.
        """
        if len(self.variables) == 0:
            print("Configure the EDP variables, before calling this function.")
            return False
        if len(self.outputs) == 0:
            print("Configure the EDP output, before calling this function.")
            return False

        for idx, pde in enumerate(pde_list):
            parameters = inspect.signature(pde).parameters
            for pde_input in parameters.keys():
                if not self.check_valid_pde_input(pde_input):
                    self.max_order = 0
                    print(f"The {idx} pde in the pde_list has invalid input: {pde_input}")
                    return False

        self.pde_list = pde_list

        return True

    def configure_boundaries(self, boundary_list: list) -> bool:
        """
        Configura las condiciones de frontera de la EDP. Esta función
        simplemente comprueba que la lista que se pasa tenga la estructura
        correcta
        """
        if len(self.variables) == 0:
            print("Configure the EDP variables, before calling this function.")
            return False

        for boundary in boundary_list:
            variables = boundary[0]
            func = boundary[1]
            output = boundary[2]

            for var in variables.keys():
                if var not in self.variables:
                    print(f"{var} is not defined as a variable.")
                    return False

            parameters = inspect.signature(func).parameters.keys()
            for var in parameters:
                if var not in self.variables:
                    print(f"{var} is not defined as a variable.")
                    return False

            if len(list(parameters) + list(variables.keys())) != len(self.variables):
                print("Invalid configuration. Parameters from function and variables in \
                the boundary should be the same as the variables defined using \
                configure_variables.")
                return False

            if output not in self.outputs.keys():
                print(f"The provided {output} output is not registered.")
                return False

        self.boundary_list = boundary_list

        return True

    def configure_train_settings(self, steps=None, alpha=None):
        """
        Configuración de las opciones del entrenamiento. Si no se pasa nada
        se utilizan las opciones por defecto
        """
        if steps is not None:
            self.steps = steps
        if alpha is not None:
            self.alpha = alpha

    def configure_points(self, interior_pts: int):
        """
        Configuración de la cantidad de puntos en el interior
        """
        self.interior_pts = interior_pts

    def start_device(self):
        """
        Inicio del dispositivo en el que se carga el aprendizaje
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_interior_points(self):
        """
        Se construyen todos los puntos
        """

        # Primero, se construyen los puntos del interior
        linspaces = [None] * len(self.variables)
        for var, idx in self.variables.items():
            range = self.var_ranges[var]
            linspaces[idx] = torch.linspace(range[0], range[1], self.interior_pts, device=self.device)

        # Luego, se construyen los puntos de la frontera
        # Si estamos en 1 variable, hay que generar más puntos de forma "artificial"
        mul = 10 if len(self.variables) == 1 else 1
        for boundary in self.boundary_list:
            variables = boundary[0]
            for var, value in variables.items():
                original = linspaces[self.variables[var]]
                pos = torch.searchsorted(original, value)
                linspaces[self.variables[var]] = torch.cat([original[:pos], torch.tensor([value]*mul, device=self.device, dtype=torch.float32), original[pos:]])
        
        grids = torch.meshgrid(*linspaces, indexing="ij")
        
        var_i = torch.stack([g.reshape(-1) for g in grids], dim=1).to(self.device)
        return var_i

    def generate_loss_graph(self, file):
        """
        Se genera un gráfico de como avanzan las funciones de pérdida
        """
        plt.plot(self.pde_losses, marker='o')
        plt.plot(self.bc_losses, marker='o')
        plt.plot(self.total_losses, marker='o')
        plt.yscale('log')
        plt.xlabel('steps')
        plt.legend(['PDE loss', 'BC loss', 'Total loss'])
        plt.title('Loss progression')
        plt.savefig(file)
        plt.close()

    def batch_derivative(self, func, x, max_order):
        """
        Se calcula las derivadas de func respecto a x hasta orden max_orden
        """
        derivatives = []
        if max_order < 1:
            return derivatives

        batch_size, dim = x.shape

        # Calcula las derivadas de orden 1 (inicial)
        deriv = autograd.functional.jacobian(lambda x: func(x).sum(), x, create_graph=True)
        derivatives.append(deriv)

        current_order = 1
        # Calcula derivadas hasta orden max_order
        while current_order < max_order:
            deriv_flat = deriv.view(batch_size, -1)
            grad_list = []
            for i in range(deriv_flat.shape[1]):
                 grad_i = autograd.grad(deriv_flat[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
                 grad_list.append(grad_i.unsqueeze(1))
            deriv = torch.cat(grad_list, dim=1)
            current_order += 1
            deriv = deriv.view(batch_size, *(dim,)*current_order)
            derivatives.append(deriv)
        
        return derivatives

    def calculate_derivative(self, derivatives, order):
        """
        Devuelve las derivadas del orden pedido
        """
        deriv = derivatives[len(order)-1]

        return deriv[(slice(None),) + tuple(order)]

    def calculate_interior_loss(self):
        """
        Calcula la pérdida en los puntos del interior de la función
        """
        # Calcula las derivadas hasta orden max_order
        pde_losses = []
        grad_var_i = self.var_i.clone().detach().requires_grad_(True)
        derivatives = self.batch_derivative(self.model, grad_var_i, self.max_order)

        # Por cada edp se calcula la pérdida
        for pde in self.pde_list:
            parameters = inspect.signature(pde).parameters.keys()
            inputs = {}
            for var in parameters:
                if var in self.variables.keys():
                    inputs[var] = self.var_i[:, self.variables[var]]
                elif var in self.outputs.keys():
                    inputs[var] = self.model(self.var_i)[:, self.outputs[var]]
                else:
                    order = []
                    for c in var[2:]:
                        order += [self.variables[c]]
                    inputs[var] = self.calculate_derivative(derivatives, order)

            pde_losses += [pde(**inputs)**2]
        
        return torch.mean(torch.sum(torch.stack(pde_losses), dim=0))

    def calculate_boundary_loss(self):
        """
        Esta función se encarga de calcular el error de
        las condiciones de frontera
        """

        boundary_losses = []
        for boundary in self.boundary_list:
            variables = boundary[0]
            function = boundary[1]
            output = boundary[2]

            # 1. Se calcula la predicción
            constraints = torch.ones(len(self.var_i), dtype=torch.bool).to(self.device)
            for var, constraint in variables.items():
                constraints &= self.var_i[:, self.variables[var]] == constraint

            prediction = self.model(self.var_i[constraints])[:, self.outputs[output]]

            # 2. Se computa el error en la frontera
            parameters = inspect.signature(function).parameters.keys()
            inputs = {}
            for var in parameters:
                inputs[var] = self.var_i[constraints][:, self.variables[var]]
            real = function(**inputs)
            
            boundary_losses += [torch.mean((prediction - real)**2)]

        return torch.sum(torch.stack(boundary_losses), dim=0)

    def calculate_loss(self):
        """
        Esta función calcula la pérdida total
        """
        interior_loss = self.calculate_interior_loss()
        boundary_loss = self.calculate_boundary_loss()
        total_loss = self.alpha * interior_loss + boundary_loss

        self.pde_losses.append(interior_loss.cpu().detach().numpy())
        self.bc_losses.append(boundary_loss.cpu().detach().numpy())
        self.total_losses.append(total_loss.cpu().detach().numpy())

        return total_loss

    def get_solution(self):
        """
        Esta función devuelve las soluciones una vez el modelo ha sido entrenado
        """
        if not self.trained:
            print("Solution cannot be generated, the model has not been trained.")
            return ()

        solutions = []
        for idx in range(len(self.outputs)):
            def sol(*args, model=self.model, idx=idx):
                new_args = []
                for arg in args:
                    new_args.append(torch.tensor(arg, device=self.device, dtype=torch.float32))
                grids = torch.meshgrid(*new_args, indexing="ij")
                var = torch.stack([g.reshape(-1) for g in grids], dim=1)
                return model(var)[:, idx].cpu().detach().numpy().reshape(grids[0].shape)

            solutions.append(sol)

        return tuple(solutions)

    def train(self, model_file, loss_graph_file="") -> bool:
        """
        Esta función realiza el entrenamiento. Nótese que la configuración ha
        tenido que ser realizada antes de llamar a esta función.
        """

        self.trained = False
        if self.device == "":
            print("Error, device has to be started before training.")
            return False
        if len(self.variables) == 0:
            print("Configure the EDP variables, before calling this function.")
            return False
        if len(self.outputs) == 0:
            print("Configure the EDP output, before calling this function.")
            return False
        if len(self.pde_list) == 0:
            print("Configure the EDPs, before calling this function.")
            return False
        if len(self.boundary_list) == 0:
            print("Configure the boundaries, before calling this function.")
            return False

        self.pde_losses = []
        self.bc_losses = []
        self.total_losses = []
        self.train_model()
        self.save_model(model_file)

        if loss_graph_file != "":
            self.generate_loss_graph(loss_graph_file)

        self.trained = True
        return True

    @abstractmethod
    def save_model(self, model_file):
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def periodic_update(self):
        pass

    def train_model(self):
        """
        Función encargada de entenar el modelo
        """
        self.var_i = self.build_interior_points()
        self.model = self.get_model()

        optimizer = LBFGS(self.model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        pbar = tqdm(range(self.steps), desc='description', ncols=100)

        for _ in pbar:
            def closure():
                optimizer.zero_grad()
                loss = self.calculate_loss()
                loss.backward()
                return loss

            if _ % 5 == 0 and _ < 50:
                self.periodic_update()

            optimizer.step(closure)