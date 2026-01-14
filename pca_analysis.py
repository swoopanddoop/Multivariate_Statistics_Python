import numpy as np
import pandas as pd

import os
os.include("stathelper.py")
import stathelper as sh

def pca_analysis(dataframe : pd.DataFrame, num_components : int):
    """
    pca_analysis
    
    :param dataframe: DataFrame containing the numerical data for PCA
    :param num_components: Number of principal components to retain
    :type num_components: int
    """

    standardized_data = dataframe - dataframe.mean()

    covariance_matrix = np.cov(standardized_data, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    principal_components = standardized_data @ selected_eigenvectors
    return principal_components, sorted_eigenvalues[:num_components]

def pca_analysis_corr(dataframe : pd.DataFrame, num_components : int):
    """
    pca_analysis_corr
    
    :param dataframe: DataFrame containing the numerical data for PCA
    :param num_components: Number of principal components to retain
    :type num_components: int
    """

    standardized_data = (dataframe - dataframe.mean()) / dataframe.std()

    correlation_matrix = np.corrcoef(standardized_data, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    principal_components = standardized_data @ selected_eigenvectors
    
    return principal_components, sorted_eigenvalues[:num_components]
