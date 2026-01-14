import numpy as np
import pandas as pd

def multiple_linear_regression(X : pd.DataFrame, y : pd.DataFrame):
    # Add a column of ones to X for the intercept term
    X_with_intercept = pd.concat([pd.DataFrame(np.ones(X.shape[0]), columns=['intercept']), X], axis=1)
    
    # Calculate the coefficients using the normal equation
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    
    return coefficients