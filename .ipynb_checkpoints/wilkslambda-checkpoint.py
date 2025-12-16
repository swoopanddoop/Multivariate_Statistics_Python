import pandas as pd
import numpy as np

def obtain_variable_matrix_from_different_samples(grouped_df_structure, sample_size : Int):
    variable_names = grouped_df_structure.columns
    num_variables = (variable_names.shape)[1]

    num_of_samples = len(grouped_df_structure.indices)
    list_of_different_variables = list()

    for i, name in enumerate(variable_names):
        variable_matrix = np.ones((num_variables,num_variables), dtype=float)

        for j, group in enumerate(grouped_df_structure):
            variable_matrix[:,j] = (group[1].to_numpy())[:,i]
            
        list_of_different_variables.append(variable_matrix)
    return (list_of_different_variables, variable_names)


    
        