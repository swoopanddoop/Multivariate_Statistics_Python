import pandas as pd
import numpy as np

import copy

import os
os.include("stathelper.py")
import stathelper as sh

def compute_wilks_lambda(E_matrix : np.array, H_matrix : np.array):
    E_determinant = np.linalg.det(E_matrix)
    EH_determinant = np.linalg.det(E_matrix + H_matrix)

    wilks_lambda = E_determinant / EH_determinant

    return wilks_lambda

def wilks_lambda_analysis(grouped_df_structure, sample_size : int):

    """
    wilks_lambda_analysis
    
    :param grouped_df_structure: Datastructure of numerical variables grouped by categorical variable
    :param sample_size: the size of every group (assumed to be the same for all groups)
    :type sample_size: int

    :return: Wilks' Lambda value
    :rtype: float
    """ 

    variable_list, variable_names = sh.obtain_variable_matrix_from_different_samples(grouped_df_structure, sample_size)
    means_matrix = sh.obtain_means_for_variable_of_each_sample(variable_list, variable_names)
    E_matrix, H_matrix = sh.obtain_E_and_H_matrices(variable_list, means_matrix)
    wilks_lambda = compute_wilks_lambda(E_matrix, H_matrix)

    return wilks_lambda
