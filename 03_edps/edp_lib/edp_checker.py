from typing import Tuple
import os
from edp_lib.edp_solver import EDPSolver
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class EDPChecker:
    def __init__(self) -> None:
        self.variables = {}
        self.var_ranges = {}
        self.outputs = []

        self.points = 400

    def configure_variables(self, variables: dict[str, Tuple]):
        """
        Configura las variables de las EDPs y su rango de definición
        """
        for idx, v in enumerate(variables.keys()):
            self.variables[v] = idx

        self.var_ranges = variables

    def configure_outputs(self, outputs: list[str]):
        """
        Configura las salidas de la EDP
        """
        self.outputs = outputs

    def create_folders(self, path: str):
        """
        Crea una carpeta si no existe.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for output in self.outputs:
            path = os.path.join(path, output)
            os.makedirs(path, exist_ok=True)

    def get_solution_from_solvers(self, solvers: dict) -> list:
        """
        Obtiene las solucioens de los modelos utilizando los mismos
        datos de entrada
        """
        solver_functions = [dict()] * len(self.outputs)

        for solver_name, solver_model in solvers.items():
            sol = solver_model.get_solution()
            if sol is None:
                return list()
            if len(sol) != len(self.outputs):
                print("Incompatibility between expected output size and solution from model.")
                return list()

            for idx in range(len(self.outputs)):
                solver_functions[idx][solver_name] = sol[idx]

        return solver_functions

    def graph_comparation_with_real(self, linspaces, var_names, output_name, results, path, aux_fn, real):
        """
        Se genera un gráfico que compara los datos con la solución real. Nótese
        que la función aux_fn es la que se utiliza según la dimensión del problema.
        """
        model_names = list(results.keys())
        num_models = len(results)
        fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 8))

        if num_models == 1:
            axes = axes.reshape(2, 1)

        for i, model_name in enumerate(model_names):
            ax = axes[0, i]
            aux_fn(ax, results[model_name], linspaces, var_names, output_name, model_name, fig, f"Solución de {model_name}" if model_name != "real" else "Solución real")

        real_result = results[real]
        for i, model_name in enumerate(model_names):
            ax = axes[1, i]
            if model_name == "real":
                fig.delaxes(ax)
                continue
            diff = real_result - results[model_name]
            aux_fn(ax, diff, linspaces, var_names, output_name, model_name, fig, f"Error de {model_name}")

        fig.subplots_adjust(wspace=0.4, hspace=0.4)

        file_name = os.path.join(path, f"comparation.pdf")
        print(f"Image save as {file_name}")
        plt.savefig(file_name, dpi=300)
        plt.clf()

    def graph_comparation_without_real(self, linspaces, var_names, output_name, results, path, aux_fn):
        """
        Se genera un gráfico que compara modelos sin comparar con una solución real. Nótese
        que cada uno de los subgráficos es generado por la función aux_fn.
        """
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
        if num_models == 1:
            axes = [axes]

        for ax, (model_name, result) in zip(axes, results.items()):
            aux_fn(ax, result, linspaces, var_names, output_name, model_name, fig, title=f"Solución de {model_name}")

        fig.subplots_adjust(wspace=0.4, hspace=0.4)

        file_name = os.path.join(path, f"comparation.pdf")
        print(f"Image save as {file_name}")
        plt.savefig(file_name, dpi=300)
        plt.clf()

    def graph(self, solutions: dict, path: str, output_name: str, aux_fn, real):
        """
        Genera un gráfico individual de la solución del modelo
        """
        linspaces = [None] * len(self.variables)
        var_names = []
        for var, idx in self.variables.items():
            var_names.append(var)
            range = self.var_ranges[var]
            linspaces[idx] = np.linspace(range[0], range[1], self.points)

        results = {}
        for model_name, solution in solutions.items():
            result = solution(*linspaces)
            results[model_name] = result

            plt.figure()
            aux_fn(plt, result, linspaces, var_names, output_name, model_name, plt, title=f"Solución de {model_name}")

            file_name = os.path.join(path, f"{model_name}.pdf")
            print(f"Image save as {file_name}")
            plt.savefig(file_name, dpi=300)
            plt.clf()
        plt.clf()

        if real == "":
            self.graph_comparation_without_real(linspaces, var_names, output_name, results, path, aux_fn)
        else:
            self.graph_comparation_with_real(linspaces, var_names, output_name, results, path, aux_fn, real)

    def graph_1D_aux(self, plot_fn, result, linspaces, var_names, output_name, model_name, fig, title="Solución"):
        """
        Función auxiliar encargada de crear gráficas de soluciones de 1 variable
        """
        plot_fn.plot(*linspaces, result)
        if "xlabel" in plot_fn.__dict__:
            plot_fn.xlabel(var_names[0])
            plot_fn.ylabel(output_name)
            plot_fn.title(f"{title}")
        else:
            plot_fn.set_xlabel(var_names[0])
            plot_fn.set_ylabel(output_name)
            plot_fn.set_title(f"{title}")
        plot_fn.grid(True)
        plot_fn.margins(x=0)

    def graph_1D(self, solutions: dict, path: str, output_name: str, real):
        """
        Genera un gráfico donde se comparan soluciones de una edp de 1 variable (EDO)
        """
        self.graph(solutions, path, output_name, self.graph_1D_aux, real)

    def graph_2D_aux(self, plot_fn, result, linspaces, var_names, output_name, model_name, fig, title="Solución"):
        """
        Función auxiliar encargada de crear gráficas de soluciones de 2 variable
        """
        x = linspaces[0]
        y = linspaces[1]
        im = plot_fn.imshow(
            result.T,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            aspect='auto',
            cmap='RdBu'
        )

        if "xlabel" in plot_fn.__dict__:
            plot_fn.xlabel(var_names[0])
            plot_fn.ylabel(var_names[1])
            plot_fn.title(f"{title}")
            fig.colorbar(im, label=f"{output_name}({var_names[0]}, {var_names[1]})")
        else:
            plot_fn.set_xlabel(var_names[0])
            plot_fn.set_ylabel(var_names[1])
            plot_fn.set_title(f"{title}")
            fig.colorbar(im, ax=plot_fn, label=f"{output_name}({var_names[0]}, {var_names[1]})")
        plot_fn.grid(True)
        plot_fn.margins(x=0)

    def graph_2D(self, solutions: dict, path: str, output_name: str, real):
        """
        Genera un gráfico donde se comparan soluciones de una edp de 2 variable
        """
        self.graph(solutions, path, output_name, self.graph_2D_aux, real)

    def create_graph(self, solvers: dict[str, EDPSolver], folder_name, real="") -> bool:
        """
        Esta función asume que los modelos ya se han entrenado. Genera gráficos
        comparando las soluciones. Si se indica quien contiene la solución real en
        la variable real se utiliza la información.
        """
        if len(self.variables) == 0:
            print("Configure the EDP variables, before calling this function.")
            return False
        if len(self.outputs) == 0:
            print("Configure the EDP output, before calling this function.")
            return False

        self.create_folders(folder_name)
        solutions = self.get_solution_from_solvers(solvers)
        if len(solutions) == 0:
            return False

        for idx in range(len(self.outputs)):
            path = os.path.join(folder_name, self.outputs[idx])
            if len(self.variables) == 1:
                self.graph_1D(solutions[idx], path, self.outputs[idx], real)
            elif len(self.variables) == 2:
                self.graph_2D(solutions[idx], path, self.outputs[idx], real)

        return True

    def get_results(self, solvers: dict[str, EDPSolver], real):
        """
        Compara diferentes modelos con la solución real y devuelve su error en la predicción.
        """
        if real not in solvers.keys():
            print(f"The provided value of real {real} should be in solvers keys.")
            return

        linspaces = [None] * len(self.variables)
        var_names = []
        for var, idx in self.variables.items():
            var_names.append(var)
            ranges = self.var_ranges[var]
            linspaces[idx] = np.linspace(ranges[0], ranges[1], self.points)

        solutions = self.get_solution_from_solvers(solvers)
        if len(solutions) == 0:
            return False

        for idx in range(len(self.outputs)):
            real_sol = solutions[idx][real]
            for model_name, solution in solutions[idx].items():
                if model_name == real:
                    continue
                diff = np.mean((real_sol(*linspaces) -  solution(*linspaces))**2)

                print(f"Total difference between {model_name} model and real solution is {diff:.10f} for function {self.outputs[idx]}.")
