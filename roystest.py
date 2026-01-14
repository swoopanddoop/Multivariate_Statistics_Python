import numpy as np
import pandas as pd

import copy

def roys_test(E_matrix : np.array, H_matrix : np.array):

    roy_matrix = np.linalg.inv(E_matrix) @ H_matrix

    eigenvalues, _ = np.linalg.eig(roy_matrix)

    roy_statistic = np.max(eigenvalues).real

    return roy_statistic