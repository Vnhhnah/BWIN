import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, jarque_bera
from statsmodels.tools import add_constant

def boekols(y, X):
    """
    Generate a regression output table including ANOVA, coefficients, and diagnostic statistics.

    Parameters:
    y (array-like): Dependent variable.
    X (array-like or DataFrame): Independent variables of any size.

    Returns:
    None: Prints the formatted regression output table.
    """
    def ensure_constant(X):
        """
        Ensures that X has a constant (intercept) column. If it doesn't, one is added.

        Parameters:
        X (array-like or DataFrame): The independent variables.

        Returns:
        X (array-like or DataFrame): X with a constant column.
        """
        if not np.any((X == 1).all(axis=0)):  # No constant column present
            X = add_constant(X)  # Add the constant column
        return X

    # Ensure that X has a constant
    X = ensure_constant(X)
    # Perform OLS regression
    model = sm.OLS(y, X).fit()

    # ANOVA components
    n = len(y)
    k = X.shape[1]
    tss = np.sum((y - np.mean(y)) ** 2, axis=0)
    rss = np.sum((model.fittedvalues - np.mean(y)) ** 2, axis=0)
    ess = np.sum(model.resid ** 2, axis=0)
    df_model = k - 1
    df_resid = n - k
    df_total = n - 1
    ms_model = rss / df_model
    ms_resid = ess / df_resid
    f_statistic = ms_model / ms_resid
    p_value_f = model.f_pvalue
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    rmse = np.sqrt(ms_resid)

    # Ensure the values are scalars if they come as Series
    tss = tss if isinstance(tss, (int, float)) else tss.item()
    df_total = df_total if isinstance(df_total, (int, float)) else df_total.item()
    tss_per_df_total = (tss / df_total) if isinstance(tss / df_total, (int, float)) else (tss / df_total).item()

    anova_table = [
        ["Source", "SS", "df", "MS"],
        ["Model", f"{rss:.3f}", f"{int(df_model)}", f"{ms_model:.3f}"],
        ["Residual", f"{ess:.3f}", f"{int(df_resid)}", f"{ms_resid:.3f}"],
        ["Total", f"{tss:.3f}", f"{int(df_total)}", f"{tss_per_df_total:.3f}"],
    ]

    # Insert the new row after the first row
    anova_table.insert(1, ["", "", "", ""])

    anova_str = "ANOVA Table:\n" + "\n".join(
        "{:<12} {:<12} {:<8} {:<12}".format(*row) for row in anova_table
    )

    # Format Regression Statistics for visual similarity
    regression_stats_table = [
        ["Stat", "Value"],  # Header row
        ["Obs", f"{n}"],
        ["F", f"{f_statistic:.3f}"],
        ["P(F)", f"{p_value_f:.3f}"],
        ["R2", f"{r_squared:.3f}"],
        ["Adj R2", f"{adj_r_squared:.3f}"],
        ["RMSE", f"{rmse:.3f}"],
    ]

    regression_stats_str = "\033[38;2;0;0;255m\033[48;2;255;255;0mModel Stats:\033[0m\n" + "\n".join(
        "{:<6} {:>8}".format(row[0], row[1]) for row in regression_stats_table
    )

    # Horizontal alignment of ANOVA and Regression Stats
    anova_lines = anova_str.split("\n")
    regression_stats_lines = regression_stats_str.split("\n")

    # Determine the width of each table
    max_anova_width = max(len(line) for line in anova_lines)
    max_regression_stats_width = max(len(line) for line in regression_stats_lines)

    # Ensure both tables have the same number of lines
    max_lines = max(len(anova_lines), len(regression_stats_lines))
    anova_lines += [""] * (max_lines - len(anova_lines))
    regression_stats_lines += [""] * (max_lines - len(regression_stats_lines))

    # Coefficients Table
    variable_names = X.columns if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X.shape[1])]
    coeff_table = [["Var", "Coef", "SE", "t", "P>|t|", "CI Lower", "CI Upper"]]
    for i, var in enumerate(variable_names):
        coeff_table.append([
            var,
            f"{model.params.iloc[i]:.3f}",    # Use .iloc for positional indexing
            f"{model.bse.iloc[i]:.3f}",      # Use .iloc for standard error
            f"{model.tvalues.iloc[i]:.3f}",  # Use .iloc for t-values
            f"{model.pvalues.iloc[i]:.3f}",  # Use .iloc for p-values
            f"{model.conf_int().iloc[i, 0]:.3f}",  # Lower confidence interval
            f"{model.conf_int().iloc[i, 1]:.3f}"   # Upper confidence interval
        ])

    coeff_table_widths = [17, 10, 8, 8, 8, 10, 10.5]  # Adjust column widths

    # Diagnostic Statistics Table
    jb_stat, jb_pval = jarque_bera(model.resid)  # Jarque-Bera statistic and p-value
    cond_no = np.linalg.cond(X)
    cond_no_str = f"{cond_no:.3f}" if cond_no <= 9999 else f"{cond_no:.1e}"

    diagnostic_table = [
        ["Omnibus:", f"{sm.stats.omni_normtest(model.resid)[0]:.3f}", "Durbin-Watson:", f"{sm.stats.durbin_watson(model.resid):.3f}"],
        ["Prob(Omnibus):", f"{sm.stats.omni_normtest(model.resid)[1]:.3f}", "Jarque-Bera (JB):", f"{jb_stat:.3f}"],
        ["Skew:", f"{skew(model.resid):.3f}", "Prob(JB):", f"{jb_pval:.3f}"],
        ["Kurtosis:", f"{kurtosis(model.resid):.3f}", "Cond. No.", cond_no_str],
    ]

    diagnostic_widths = [20, 20, 19, 18]  # Adjust column widths for Diagnostic Table

    # Print Outputs
    print("------------------------------------------------------------------")
    print("\033[38;2;255;165;0m        Regression Output Designed for BOEK Students Only!        \033[0m")
    print("------------------------------------------------------------------")

    # Print ANOVA Table and Regression Stats Horizontally
    for anova_line, reg_line in zip(anova_lines, regression_stats_lines):
        print(f"{anova_line:<{max_anova_width}}    {reg_line}")

    # Print Coefficients Table
    print("\033[38;2;255;192;203m\033[48;2;50;50;50mCoefficients Table:\033[0m")
    print("".join(f"{header:<{width}}" for header, width in zip(coeff_table[0], coeff_table_widths)))
    print("-" * 66)
    for row in coeff_table[1:]:
        print("".join(f"{col:<{width}}" for col, width in zip(row, coeff_table_widths)))

    # Print Diagnostic Statistics Table
    print("\033[38;2;255;255;0m\033[48;2;0;0;255mDiagnostic Statistics:\033[0m")
    print("------------------------------------------------------------------")
    for row in diagnostic_table:
        print("".join(f"{col:<{width}}" for col, width in zip(row, diagnostic_widths)))

    print("------------------------------------------------------------------")
    # If return_model is True, return the model object
    return model
