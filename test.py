import os
print(os.listdir(r'C:\Users\trang tran\Documents\Day 3-20250923\FRED'))
import sys
# Add the directory to sys.path
sys.path.append(r'C:\Users\trang tran\Documents\Day 3-20250923\FRED')
# Set working directory
os.chdir(r'C:\Users\trang tran\Documents\Day 3-20250923\FRED')
import pandas as pd
from boek_reg import boekols

df = pd.read_excel('FRED Data for Vietnam.xls', sheet_name='Data')

file_path = r'FRED Data for Vietnam.xls'
df = pd.read_excel(file_path, sheet_name='Data')

"""
df2 = pd.read_excel('FRED Data for Vietnam.xls', sheet_name='Data')
"""
print(df.columns)

#Exploring the variables:
df['H'].describe()  # Provides summary statistics for the column.
df['H'].mean()      #Computes the mean.
df['H'].median()    # Computes the median.
df['H'].mode()      # Returns the mode(s).
df['H'].std()       # Calculates the standard deviation.
df['H'].var()       # Calculates the variance.
df['H'].sum()       # Computes the sum of all values.
df['H'].min()       # Returns the minimum value.
df['H'].max()       # Returns the maximum value.
from boek_tab import tab
tab(df,'H')
from boek_sumd import sumd
sumd(df,'H',3)
#graphics
import matplotlib.pyplot as plt
# Plotting a histogram for column 'H'
df['H'].plot(kind='hist', title='Histogram of H', edgecolor='black')
plt.xlabel('H')
plt.ylabel('Frequency')
plt.show()

# Plotting a density plot (KDE) for column 'H'
df['H'].plot(kind='density', title='Kernel Density Estimate of H')
plt.xlabel('H')
plt.ylabel('Density')
plt.show()

# Plotting a box plot for column 'H'
df['H'].plot(kind='box', title='Box Plot of H')
plt.ylabel('H')
plt.show()
#here you examine correlation between one variable and the rest
correlation = df.corr()['H'].sort_values(ascending=False)
import seaborn as sns
# Display the correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # Add annotations for clarity
plt.title("Correlation Heatmap")
plt.show()

#Generating new variables

import numpy as np
# Creating new columns as the natural log of K, Y, H, and L
# It is a standard practice in econometrics that we take the log of the variables
# which have large values, large variations to transform variables into variables
# with PDF look like a bell (the normal distribution)
df['log_K'] = np.log(df['K'])
df['log_Y'] = np.log(df['Y'])
df['log_H'] = np.log(df['H'])
df['log_L'] = np.log(df['L'])
# Display the updated DataFrame
print(df.head())

#First regression model:
import statsmodels.api as sm
# Define the independent variables (log_K, log_H, log_L) and add a constant
X = df[['log_K', 'log_H', 'log_L']]
X = sm.add_constant(X)  # Adds a constant term (intercept) to the model

# Define the dependent variable (log_Y)
y = df['log_Y']

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())

boekols(y,X)
## Compute growth rates
df = df.sort_values(by='year')  # Ensure sorted by year
df['growth_log_K'] = df['log_K'].diff()
df['growth_log_Y'] = df['log_Y'].diff()
df['growth_log_H'] = df['log_H'].diff()
df['growth_log_L'] = df['log_L'].diff()

# Define the independent variables (growth rates of K, H, and L)
X2 = df[['growth_log_K', 'growth_log_H', 'growth_log_L']]
X2 = sm.add_constant(X2)  # Adds a constant term (intercept) to the model
X2 = X2.dropna()
# Define the dependent variable (growth rate of Y)
y2 = df['growth_log_Y'].dropna()


# Fit the OLS regression model
model2 = sm.OLS(y2, X2, missing='drop').fit()  # `missing='drop'` ensures rows with NaNs are ignored

# Print the regression results
print(model2.summary())

#Practice: We are now studying the first dataset used in regression analysis
#The GaltonFamilies file
"""
1. Import this file to python
2. Explore each variable using the following command: .describe(), .mean(),
    .median(), .mode(), .std(), .var(), .sum(), .min() and .max()
3. Graph the kernel density, the historgram and the box plot of each variable.
4. Run a regrssion model explaining the variation of Height on "father" and "mother"
"""
file_path = r'GaltonFamilies.xlsx'
df_galton = pd.read_excel(file_path, sheet_name='GaltonFamilies')

"""
missing optional dependency 'openpyxl'.  Use pip or conda to install openpyxl.
pip install openpyxl
"""
# Assign new column names
df_galton.columns = ['id', 'family', 'father', 'mother', 'midparentHeight', 'children', 'childNum', 'gender', 'Height']
X3 = df_galton[['father', 'mother', 'midparentHeight', 'children', 'childNum']].dropna()
y3 = df_galton[['Height']]
# Optional: Display the first few rows to verify the changes
print(df_galton.head())

from boek_tab import tab
tab(df_galton, 'childNum')

boekols(y3, X3)
