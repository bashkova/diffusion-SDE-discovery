import logging
import re
import itertools
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import epde
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EquationProcessor:
    """Class for processing and organizing equations discovered by EPDE."""

    def __init__(self):
        self.regex = re.compile(r', freq:\s\d\S\d+')

    @staticmethod
    def dict_update(d_main: Dict, term: str, coeff: float, k: int) -> Dict:
        """UPDATING DICTIONARIES FOR EQUATIONS"""
        str_t = '_r' if '_r' in term else ''
        arr_term = re.sub('_r', '', term).split(' * ')

        perm_set = list(itertools.permutations(range(len(arr_term))))
        structure_added = False

        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_t
            if temp in d_main:
                if k - len(d_main[temp]) >= 0:
                    d_main[temp] += [0 for _ in range(k - len(d_main[temp]))] + [coeff]
                else:
                    d_main[temp][-1] += coeff
                structure_added = True

        if not structure_added:
            d_main[term] = [0 for _ in range(k)] + [coeff]

        return d_main

    def equation_table(self, k: int, equation, dict_main: Dict, dict_right: Dict) -> List[Dict]:
        """CREATING EQUATION TABLES"""
        equation_s = equation.structure
        equation_c = equation.weights_final
        text_form_eq = self.regex.sub('', equation.text_form)

        flag = False
        for t_eq in equation_s:
            term = self.regex.sub('', t_eq.name)
            for t in range(len(equation_c)):
                c = equation_c[t]
                if f'{c} * {term} +' in text_form_eq:
                    dict_main = self.dict_update(dict_main, term, c, k)
                    equation_c = np.delete(equation_c, t)
                    break
                elif f'+ {c} =' in text_form_eq:
                    dict_main = self.dict_update(dict_main, "C", c, k)
                    equation_c = np.delete(equation_c, t)
                    break
            if f'= {term}' == text_form_eq[text_form_eq.find('='):] and not flag:
                flag = True
                dict_main = self.dict_update(dict_main, term, -1., k)

        return [dict_main, dict_right]

    def object_table(self, res: List, variable_names: List[str],
                     table_main: List[Dict], k: int, title: str) -> Tuple[List[Dict], int]:
        """CREATING OBJECT FOR TABLES"""

        def filter_func(*args, **kwargs):
            return True

        for list_SoEq in res:
            for SoEq in list_SoEq:
                if filter_func(SoEq, variable_names):
                    for n, value in enumerate(variable_names):
                        gene = SoEq.vals.chromosome.get(value)
                        table_main[n][value] = self.equation_table(
                            k, gene.value, *table_main[n][value]
                        )
                    k += 1
        return table_main, k

    def preprocessing_table(self, variable_name: List[str],
                            table_main: List[Dict], k: int) -> pd.DataFrame:
        """PREPROCESSING FOR CREATING DATAFRAME"""
        data_frame_total = pd.DataFrame()

        for dict_var in table_main:
            for var_name, list_structure in dict_var.items():
                general_dict = {}
                for structure in list_structure:
                    general_dict.update(structure)
                dict_var[var_name] = general_dict

        for dict_var in table_main:
            for var_name, general_dict in dict_var.items():
                for key, value in general_dict.items():
                    if len(value) < k:
                        general_dict[key] = value + [0. for _ in range(k - len(value))]

        data_frame_main = [{i: pd.DataFrame()} for i in variable_name]

        for n, dict_var in enumerate(table_main):
            for var_name, general_dict in dict_var.items():
                data_frame_main[n][var_name] = pd.DataFrame(general_dict)

        for n, var_name in enumerate(variable_name):
            data_frame_temp = data_frame_main[n].get(var_name).copy()
            list_columns = [f'{col}_{var_name}' for col in data_frame_temp.columns]
            data_frame_temp.columns = list_columns
            data_frame_total = pd.concat([data_frame_total, data_frame_temp], axis=1)

        return data_frame_total


def epde_multisample_discovery(t_ax, variables, diff_method: str = 'poly', find_simple_eqs: bool = False):
    """Executes the EPDE multisample discovery process."""
    samples = [[t_ax[i], [var[i] for var in variables]] for i in range(len(t_ax))]
    epde_search_obj = epde.EpdeMultisample(data_samples=samples, use_solver=False, boundary=0,
                                           verbose_params={'show_iter_idx': True})
    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', preprocessor_kwargs={'epochs_max': 1})
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': True, 'sigma': 1,
                                                              'polynomial_window': 7, 'poly_order': 4})
    epde_search_obj.set_moeadd_params(population_size=8, training_epochs=1)
    variable_names = [f'u{i}' for i in range(len(variables))]

    if not find_simple_eqs:
        logger.info("Searching for complex equations")
        factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}
        epde_search_obj.fit(
            samples=samples, variable_names=variable_names,
            max_deriv_order=(2,), equation_terms_max_number=5,
            data_fun_pow=2, deriv_fun_pow=2,
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-5, 1e-3)
        )
    else:
        logger.info("Searching for linear equations")
        epde_search_obj.fit(
            samples=samples, variable_names=variable_names,
            max_deriv_order=(1,), equation_terms_max_number=10,
            data_fun_pow=1, deriv_fun_pow=1,
            equation_factors_max_number={'factors_num': [1], 'probas': [1.0]},
            eq_sparsity_interval=(1e-4, 1e-1)
        )
    return epde_search_obj


def run_discovery_and_save(t_axis, x_vars, is_simple_search: bool, output_dir: Path):
    """
     EPDE discovery for a specific search type
    """
    if is_simple_search:
        logger.info("\n--- Running Simple Equations Search ---")
        csv_filename = 'discovered_equations_simple.csv'
    else:
        logger.info("\n--- Running Complex Equations Search ---")
        csv_filename = 'discovered_equations_complex.csv'

    output_path = output_dir / csv_filename

    epde_result = epde_multisample_discovery(
        t_axis, x_vars, diff_method='poly', find_simple_eqs=is_simple_search
    )

    results = epde_result.equations(False)
    print(results)

    if not results:
        logger.warning("No valid equations were discovered by EPDE. Skipping file save.")
        return

    logger.info("Equations found. Processing for saving.")

    valid_results = [res_list for res_list in results if res_list]
    if not valid_results:
        logger.warning("Equation results were empty after filtering. Skipping file save.")
        return


    num_vars = len(x_vars)
    variable_names = [f'u{i}' for i in range(num_vars)]
    logger.info("Equations found. Processing for saving.")
    processor = EquationProcessor()


    table_main = [{i: [{}, {}]} for i in variable_names]
    k = 0
    table_main, k = processor.object_table(results, variable_names, table_main, k, '')
    equations_df = processor.preprocessing_table(variable_names, table_main, k)
    equations_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
    logger.info(f"Equations successfully saved to: {output_path}")


if __name__ == "__main__":
    file_path = Path(r'C:\Users\Ksenia\NSS\ODE_projects\diffSDE\data')
    output_dir = file_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = np.load(file_path / 'trajectories-2')
        data = data[:, :21, :]
        print(f"Loaded data with shape: {data.shape}")

        num_trajectories = 5
        num_timesteps = data.shape[1]
        num_vars = data.shape[2]

        trajectories = [data[i] for i in range(num_trajectories)]
        t_axis = [[np.linspace(0, 1, num_timesteps)] for _ in range(num_trajectories)]
        x_vars = [[traj[:, i] for traj in trajectories] for i in range(num_vars)]

        # Запуск
        run_discovery_and_save(t_axis, x_vars, is_simple_search=True, output_dir=output_dir)

    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path / 'trajectories-2'}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
