import pandas as pd
import statsmodels.api as sm
import numpy as np

def vif_for_variable(X, j):
    """
    Compute the Variance Inflation Factor (VIF) for the j-th column of X.
    X should already include the constant column if desired.
    """
    # target variable = column j
    y = X[:, j]
    # predictors = all other columns
    X_others = np.delete(X, j, axis=1)
    
    # regress y on the other columns
    model = sm.OLS(y, sm.add_constant(X_others)).fit()
    
    R2 = model.rsquared
    if R2 == 1.0:
        return np.inf
    else:
        return 1.0 / (1.0 - R2)

def check_vif(X, threshold=10.0):
    """
    Compute Variance Inflation Factors (VIF) for a set of regressors.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame with only the explanatory variables (no dependent variable).
    threshold : float
        VIF values above this threshold will trigger a warning.

    Returns
    -------
    vif_df : pandas.DataFrame
        Table of variables and their VIF values.
    """
    # add constant for intercept
    X_const = sm.add_constant(X)
    
    vif_data = []
    for j in range(X_const.shape[1]):
        vif_val = vif_for_variable(X_const.values, j)
        vif_data.append((X_const.columns[j], vif_val))
    
    vif_df = pd.DataFrame(vif_data, columns=["Variable", "VIF"])
    print(vif_df)
    
    # warnings, similar to Stata’s behavior
    for var, val in vif_data:
        if np.isinf(val) or val > threshold:
            print(f"⚠️ Warning: '{var}' has VIF={val:.2f} (possible multicollinearity).")
    
    return vif_df
