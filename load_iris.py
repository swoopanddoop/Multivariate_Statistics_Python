import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.system("wilkslambda.py")
import wilkslambda as wl


#Load the dataset
df = pd.read_csv('iris.csv')

#Groupby

df_grouped = df.groupby("variety")

#Perform Wilks' Lambda Analysis
sample_size = 50
wilks_lambda_value = wl.wilks_lambda_analysis(df_grouped, sample_size)
print("Wilks' Lambda Value: ", wilks_lambda_value)

