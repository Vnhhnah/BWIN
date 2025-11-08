import pandas as pd
import numpy as np

def summarize(data, *variables):
    """
    Summarize variables in a DataFrame, compute statistics, and apply formatting.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - *variables: Column names as individual strings or a single string.
    
    Returns:
    - pd.DataFrame: A formatted summary table with blanks for NaN values.
    """
    variables = list(variables)

    def format_value(value, is_integer=False):
        """Format a single value: integers without decimals, floats with two decimals, blanks for NaN."""
        if pd.isna(value):  # Replace NaN with a blank
            return ""
        if is_integer:
            return int(value)  # Ensure integers are displayed without decimals
        elif isinstance(value, (float, np.float64)):
            rounded_value = round(value, 2)  # Round the value to two decimals
            if rounded_value.is_integer():  # Check if it is effectively an integer
                return int(rounded_value)  # If so, return as integer without decimals
            return rounded_value  # Otherwise, return the float rounded to 2 decimal places
        return value

    def safe_stat(func, var):
        """Safely compute a statistic."""
        # Replace . and empty strings with NaN and convert to numeric
        data[var] = data[var].replace({'.': np.nan, '': np.nan})
        data[var] = pd.to_numeric(data[var], errors='coerce')  # Convert to numeric, setting errors as NaN
        if pd.api.types.is_numeric_dtype(data[var]):
            return func(data[var])
        return np.nan

    # Compute the raw statistics
    raw_summary = pd.DataFrame({
        'Variable': variables,
        'Obs': [data[var].notna().sum() for var in variables],  # Count non-NaN values
        'Mean': [safe_stat(pd.Series.mean, var) for var in variables],
        'Median': [safe_stat(pd.Series.median, var) for var in variables],
        'Std. Dev.': [safe_stat(pd.Series.std, var) for var in variables],
        'Min': [safe_stat(pd.Series.min, var) for var in variables],
        'Max': [safe_stat(pd.Series.max, var) for var in variables],
    })

    # Format values directly while creating raw summary
    for col in ['Obs', 'Mean', 'Median', 'Std. Dev.', 'Min', 'Max']:
        # Special handling for 'Obs' column to ensure integer formatting
        if col == 'Obs':
            raw_summary[col] = raw_summary[col].apply(lambda x: format_value(x, is_integer=True))
        else:
            raw_summary[col] = raw_summary[col].apply(lambda x: format_value(x))

    return raw_summary

# Alias for quick access
sum = summarize
