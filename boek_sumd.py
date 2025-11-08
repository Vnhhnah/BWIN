from scipy.stats import skew, kurtosis
import numpy as np 

# Define the function with an additional parameter for the number of decimals
def sumd(data, variable, decimals=2):
    """
    Summarizes the given variable in the DataFrame similar to the Stata output.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    variable (str): The column name of the variable to summarize.
    decimals (int, optional): Number of decimal places to format the output. Default is 2.
    
    Returns:
    str: A formatted summary table as a string.
    """
    # Drop missing values
    clean_data = data[variable].dropna()

    # Calculate percentiles
    percentiles = {
        '1%': np.percentile(clean_data, 1),
        '5%': np.percentile(clean_data, 5),
        '10%': np.percentile(clean_data, 10),
        '25%': np.percentile(clean_data, 25),
        '50%': np.percentile(clean_data, 50),
        '75%': np.percentile(clean_data, 75),
        '90%': np.percentile(clean_data, 90),
        '95%': np.percentile(clean_data, 95),
        '99%': np.percentile(clean_data, 99),
        '99.9%': np.percentile(clean_data, 99.9),
    }
    
    # Basic statistics
    mean_value = clean_data.mean()
    std_dev = clean_data.std()
    variance = clean_data.var()
    skewness = skew(clean_data)
    kurt = kurtosis(clean_data)
    obs = len(clean_data)
    missing_values = data[variable].isna().sum()
    
    # Smallest and largest values
    smallest_values = clean_data.nsmallest(5).values
    largest_values = clean_data.nlargest(5).values
    
    # Organizing the output
    summary_table = f"""
    Total {variable}

    Percentiles    Smallest
     1%    {percentiles['1%']:.{decimals}f}          {smallest_values[0]:.{decimals}f}
     5%    {percentiles['5%']:.{decimals}f}          {smallest_values[1]:.{decimals}f}
    10%    {percentiles['10%']:.{decimals}f}          {smallest_values[2]:.{decimals}f}
    25%    {percentiles['25%']:.{decimals}f}          {smallest_values[3]:.{decimals}f}
    50%    {percentiles['50%']:.{decimals}f}          {smallest_values[4]:.{decimals}f}

                             Obs    {obs:10}
                             Mean   {mean_value:.{decimals}f}
                             Std. dev.   {std_dev:.{decimals}f}
                             Missing   {missing_values:10}

    Largest
    75%    {percentiles['75%']:.{decimals}f}          {largest_values[4]:.{decimals}f}
    90%    {percentiles['90%']:.{decimals}f}          {largest_values[3]:.{decimals}f}
    95%    {percentiles['95%']:.{decimals}f}          {largest_values[2]:.{decimals}f}
    99%    {percentiles['99%']:.{decimals}f}          {largest_values[1]:.{decimals}f}
    99.9%  {percentiles['99.9%']:.{decimals}f}          {largest_values[0]:.{decimals}f}

    Variance  {variance:.{decimals}f}
    Skewness  {skewness:.{decimals}f}
    Kurtosis  {kurt:.{decimals}f}
    """
    
    return print(summary_table)

