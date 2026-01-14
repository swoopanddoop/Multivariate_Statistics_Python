import numpy as np 
import pandas as pd

def pillais_trace_test(E_matrix : np.array, H_matrix : np.array):
    """
    pillais_trace_test

    :param E_matrix: Error matrix
    :type E_matrix: np.array
    :param H_matrix: Hypothesis matrix
    :type H_matrix: np.array
    :return: Pillai's Trace statistic
    :rtype: float
    """ 

    inv_E = np.linalg.inv(E_matrix)
    product_matrix = inv_E @ H_matrix

    eigenvalues, _ = np.linalg.eig(product_matrix)

    pillais_trace = np.sum(eigenvalues.real / (1 + eigenvalues.real))

    return pillais_trace