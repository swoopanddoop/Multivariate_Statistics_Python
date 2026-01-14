import numpy as np
import copy

"""Statistical Helper Functions"""


"""

Functions for samples grouped by categorical variable

"""
def obtain_variable_matrix_from_different_samples(grouped_df_structure, sample_size : int):
    grouped_list = list(grouped_df_structure)

    variable_names_list = list(grouped_list[0][1].columns)
    variable_names = variable_names_list[:-1]

    num_variables = len(variable_names)
    number_of_samples = len(grouped_df_structure)

    list_of_different_variables = list()

    for i in range(num_variables):
        variable_matrix = np.ones((sample_size, number_of_samples), dtype=float)

        for j, group in enumerate(grouped_df_structure):
            variable_matrix[:,j] = group[1].to_numpy()[:,i]

        list_of_different_variables.append(variable_matrix)
    
    return (list_of_different_variables, variable_names)

def obtain_means_for_variable_of_each_sample(variable_list : list, variable_names):
    num_samples = len(variable_list[0][0])
    num_variables = len(variable_names)
    means_matrix = np.ones((num_samples, num_variables), dtype=float)

    for i in range(num_samples):
        for j in range(num_variables):
            means_matrix[i,j] = np.mean(variable_list[j][:,i])

    return means_matrix

def obtain_E_and_H_matrices(variable_list : list, means_matrix : np.array):
    num_variables = len(variable_list)
    num_samples = (means_matrix.shape[0])

    E_matrix = np.zeros((num_variables,num_variables), dtype=float)
    H_matrix = np.zeros((num_variables,num_variables), dtype=float)

    overall_mean_vector = np.mean(means_matrix, axis=0)

    centered_list = copy.deepcopy(variable_list)

    for i, variable_matrix in enumerate(centered_list):
        centered_list[i] = variable_matrix - means_matrix[:,i].T

    for i in range(num_variables):
        for j in range(num_variables):
            for k in range(num_samples):
                E_matrix[i,j] += np.dot(centered_list[i][:,k], centered_list[j][:,k])
                H_matrix[i,j] += np.dot((means_matrix[k,i] - overall_mean_vector[i]), (means_matrix[k,j] - overall_mean_vector[j])) * variable_list[0].shape[0]

    return (E_matrix, H_matrix)

"""

Functions for one sample

"""


