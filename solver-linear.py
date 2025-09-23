import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import numpy as np
import pandas as pd

import epde
from epde.integrate.pinn_integration import SolverAdapter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EquationAnalyzer:

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()

    def load_equation_data(self, filename: str) -> bool:
        try:
            file_path = self.data_path / 'output' / filename
            if not file_path.exists():
                logger.error(f"data file not found: {file_path}")
                return False

            self.df = pd.read_csv(file_path).fillna(0)
            logger.info(f"loaded data from {filename} with shape {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"error loading equation data from {filename}: {e}")
            return False

class EquationConstructor:
    @staticmethod
    def _create_equation_terms(row: pd.Series, var_names: List[str]) -> Dict[str, Dict]:
        terms = {}
        num_vars = len(var_names)

        for eq_idx, eq_var in enumerate(var_names):

            # constant
            col_name_c = f'C_{eq_var}'
            if col_name_c in row:
                terms[f'C_{eq_var}'] = {
                    'coeff': row[col_name_c],
                    'term': [[None]],
                    'pow': [0],
                    'var': [0]
                }

            for term_idx in range(num_vars):  # u_i or du_i/dt

                col_name_var = f'u{term_idx}_{eq_var}'
                if col_name_var in row:
                    terms[col_name_var] = {
                        'coeff': row[col_name_var],
                        'term': [[None]],
                        'pow': [1],
                        'var': [term_idx]
                    }

                # derivative du0/dx0, du1/dx0
                col_name_deriv = f'du{term_idx}/dx0_{eq_var}'
                if col_name_deriv in row:
                    terms[col_name_deriv] = {
                        'coeff': row[col_name_deriv],
                        'term': [[0]],
                        'pow': [1],
                        'var': [term_idx]
}
        return terms

    def construct_equation_systems(self, coefficients_df: pd.DataFrame, var_names: List[str]) -> List[Dict]:
        systems = []
        coefficients_df.columns = coefficients_df.columns.str.replace(r'{power: 1.0}', '', regex=False)

        for _, row in coefficients_df.iterrows():
            full_terms = self._create_equation_terms(row, var_names)

            system_equations = []
            for eq_var in var_names:
                eq_dict = {k: v for k, v in full_terms.items() if k.endswith(f'_{eq_var}') and v['coeff'] != 0}
                system_equations.append(eq_dict)
            if any(system_equations):
                systems.append(system_equations)
            else:
                logger.warning(" all zero coefficients, skipping ")
        return systems


class ODESolver:
    """Solves ODE systems using EPDE."""
    def __init__(self, params: Dict):
        self.adapter = SolverAdapter()
        for key, value in params.items():
            self.adapter.change_parameter(key, value)
        logger.info(f"Solver configured with parameters: {params}")

    @staticmethod
    def create_boundary_operator(key: str, var_idx: int, grid_loc: float, value: float) -> Dict[str, Any]:
        bop = epde.integrate.BOPElement(axis=0, key=key, term=[None], power=1, var=var_idx)
        bop.set_grid(torch.tensor([[grid_loc]], dtype=torch.float32))
        bop.values = torch.tensor([[value]], dtype=torch.float32)
        return {
            'type': 'boundary',
            'bnd_loc': torch.Tensor([bop.location]),
            'bnd_op': {bop.operator_form[0]: bop.operator_form[1]},
            'bnd_val': bop.values
        }

    def solve_system(self, equations: List[Dict], grid: np.ndarray, bcs: List[Dict]) -> np.ndarray | None:
        try:
            if grid.size == 0:
                logger.error("Empty grid provided to solver.")
                return None
            if not bcs:
                logger.error("No valid boundary conditions provided.")
                return None

            _, solution = self.adapter.solve_epde_system(
                system=equations,
                grids=[torch.from_numpy(grid).float()],
                boundary_conditions=bcs,
                mode='NN',
                to_numpy=True,
                grid_var_keys=['t']
            )
            return solution

        except Exception as e:
            logger.error(f"Error solving ODE system: {e}")
            logger.error(traceback.format_exc())
            return None

class SolutionManager:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def save_array(self, solution: np.ndarray, base_filename: str) -> None:
        npy_path = self.output_path / f"{base_filename}.npy"
        np.save(npy_path, solution)
        logger.info(f"solution saved to: {npy_path}")


def average_solutions(solutions_list: List[np.ndarray]) -> np.ndarray | None:
    if not solutions_list:
        return None
    valid_solutions = [s for s in solutions_list if s is not None and s.ndim == 2 and s.shape[0] > 0]
    if not valid_solutions:
        return None
    return np.mean(np.array(valid_solutions), axis=0)


def main():

    base_path = Path(r'C:\Users\Ksenia\NSS\ODE_projects\diffSDE\data')
    equation_csv_file = 'discovered_equations_simple.csv'
    trajectory_data_file = 'trajectories-2.npy'  #for initial conditions
    output_filename_base = 'solved_trajectory'

    num_vars = 5
    variable_names = [f'u{i}' for i in range(num_vars)]

    solver_params = {
        'epochs': 5000,
        'learning_rate': 1e-4,
        'hidden_layers_nodes': [64, 64],
        'verbose': True,
        'optimizer': 'Adam' #LBFGS,
        # 'activation': torch.nn.SiLU
    }

    t_end = 1.0
    t_points = 100
    t_grid = np.linspace(0, t_end, t_points).reshape(-1, 1)

    analyzer = EquationAnalyzer(base_path)
    if not analyzer.load_equation_data(equation_csv_file):
        return

    constructor = EquationConstructor()
    equation_systems = constructor.construct_equation_systems(analyzer.df, variable_names)

    if not equation_systems:
        logger.error("no systems were constructed")
        return
    logger.info(f"constructed {len(equation_systems)} equation system(s).")
    print(equation_systems)

    try:
        full_data = np.load(base_path / trajectory_data_file)
        initial_conditions = full_data[0, 0, :]
        if initial_conditions.shape[0] != num_vars:
            raise ValueError("no expected number of variables.")
        logger.info(f"using initial conditions from data: {initial_conditions}")
    except Exception as e:
        logger.error(f"could not load initial conditions from {trajectory_data_file}: {e}")
        logger.warning("using default initial conditions of 0.5 for all variables.")
        initial_conditions = np.full(num_vars, 0.5)

    solver = ODESolver(params=solver_params)
    solution_manager = SolutionManager(base_path / 'output')
    all_solutions = []

    for system_idx, system_to_solve in enumerate(equation_systems):
        logger.info(f"\n=== Solved SYSTEM {system_idx + 1} ===")

        # initial conditions at t=0
        boundary_conditions = []
        for i in range(num_vars):
            bc = solver.create_boundary_operator(key=f'u{i}', var_idx=i, grid_loc=0.0, value=initial_conditions[i])
            boundary_conditions.append(bc)

        logger.info("Solving")
        solution = solver.solve_system(system_to_solve, t_grid, boundary_conditions)

        if solution is not None:
            logger.info(f"Successfully solved system #{system_idx + 1}. Solution shape: {solution.shape}")
            all_solutions.append(solution)
            solution_manager.save_array(solution, f"{output_filename_base}_system_{system_idx}")
        else:
            logger.error(f"Failed to solve system #{system_idx + 1}")

    # Average solution
    if all_solutions:
        if len(all_solutions) > 1:
            logger.info("averaging solutions from systems")
            avg_solution = average_solutions(all_solutions)
            if avg_solution is not None:
                solution_manager.save_array(avg_solution, f"{output_filename_base}_averaged")


if __name__ == "__main__":
    main()