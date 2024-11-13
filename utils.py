
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import math

def missing_values_plot(data: pd.DataFrame, target: str):
    
    """
    Plots the ratio of missing to available values for each feature in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        target (str): The target column name used to filter usable samples.

    Raises:
        ValueError: If `target` is not in `data.columns`.
    """
    

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    # Filter usable samples where target is not null
    usable = data[data[target].notnull()]
    if usable.empty:
        print("No usable samples found (all target values are missing).")
        return
    
    print(usable.shape)


    # Calculate missing value count and ratio for each feature
    missing_count = usable.isnull().sum().reset_index()
    missing_count.columns = ['feature', 'null_count']
    missing_count['null_ratio'] = missing_count['null_count'] / len(usable)

    # Filter features with missing values only
    #missing_count = missing_count[missing_count['null_count'] > 0]
    
    if missing_count.empty:
        print("No missing values found in the usable samples.")
        return

    # Sort by missing count for better readability in plot
    missing_count = missing_count.sort_values('null_count', ascending=False)

    # Plot the missing vs available values as a horizontal bar chart
    plt.figure(figsize=(10, max(15, len(missing_count) * 0.3)))
    plt.title(f'Missing values over the {len(usable)} usable samples')
    
    plt.barh(np.arange(len(missing_count)), 
             missing_count['null_ratio'], 
             color='red', label='missing')
    
    plt.barh(np.arange(len(missing_count)), 
             1 - missing_count['null_ratio'],
             left=missing_count['null_ratio'],
             color='gray', label='available')
    
    for i, (null_count, null_ratio) in enumerate(zip(missing_count['null_count'], missing_count['null_ratio'])):
        if null_ratio > 0.05: 
            plt.text(null_ratio - 0.02, i, f'{null_count}', color='white', va='center', ha='right')
        else:
            plt.text(null_ratio + 0.02, i, f'{null_count}', color='black', va='center', ha='left')

    
    plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlim(0, 1)
    plt.xlabel("Percentage")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    




def plot_correlation_matrix(data: pd.DataFrame, target: str, method: str = 'pearson'):
    
    """
    Plots the correlation matrix of the features in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        target (str): The target column name used to filter usable samples.
        threshold (float): The threshold for filtering the correlation matrix.

    Raises:
        ValueError: If `target` is not in `data.columns`.
    """
    
    if target is object:
        raise ValueError(f"Target column '{target}' is not numeric.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    
     # Filter data to include only numeric columns
    
    data = data[data[target].notnull()]

    usable = data.select_dtypes(include=[np.number])
    num_cols = len(usable.columns)
    


    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("Invalid correlation method. Choose from 'pearson', 'spearman', or 'kendall'.")


    if usable.empty:
        print("No usable samples found (all target values are missing).")
        return
    
    #Calculate the correlation matrix

    corr = usable.corr(method = method)
    corr = corr.dropna(how='all').dropna(axis=1, how='all')

    #Plot the correlation matrix as a heatmapp
    plt.figure(figsize=(max(0.5 * num_cols, 10), max(0.5 * num_cols, 10)))
    plt.title(f'Correlation matrix over the {len(usable)} usable samples with {method} correlation')

    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.color_palette("rocket", as_cmap=True), center=0)
    plt.tight_layout()
    plt.show()
    


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
import warnings

def histplot_all(data: pd.DataFrame, target: str, hue: bool = True, log_scale: bool = False):
    """
    Plots histograms for all numeric columns in a DataFrame with optional class separation by hue
    and optional log scaling using log1p for the x-axis.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - target (str): The target column name, used as hue if it's categorical and hue is True.
    - hue (bool): If True, will use target as hue for categorical targets; default is True. Max Hue is limited to 5.
    - log_scale (bool): If True, applies a log1p scale to the x-axis for each histogram, default is False.
    """
    #Check if the target column exists in the DataFrame
    if target not in data.columns:
        raise ValueError(f"The target column '{target}' does not exist in the provided DataFrame.")
    
    #Check if hue is to be used (only if the target is categorical and hue is True)
    use_hue = hue and (data[target].dtype == 'object' or data[target].nunique() < 5)
    
    #Filter numeric columns excluding the target
    numeric_cols = [col for col in data.columns if data[col].dtype != object and col != target]
    
    #Determine the number of rows and columns for the grid
    n_cols = 3  
    n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate rows based on columns and the number of plots
    
    #Set up the matplotlib figure with the calculated grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        #Apply log1p transformation if log_scale is True
        if log_scale:
            #Check for non-positive values in the data
            if (data[col] < 0).any():
                warnings.warn(f"Column '{col}' contains negative values and cannot use log scale. Plotting without log scale.")
                plot_data = data[col]  #Use original data if log scale isn't possible due to negative values
                xlabel = col
                title = f"{col} Distribution"
            else:
                plot_data = np.log1p(data[col])  #Log1p-transform the data
                xlabel = f'Log1p of {col}'  # Label for log-transformed x-axis
                title = f'Log1p Transformed {col} Distribution'
        else:
            plot_data = data[col]
            xlabel = col
            title = f"{col} Distribution"

        #Plot with or without hue depending on the use_hue flag
        if use_hue:
            sns.histplot(data=data.assign(**{col: plot_data}), x=col, kde=True, hue=target, ax=axes[i], multiple="stack")
        else:
            sns.histplot(data=plot_data, kde=True, ax=axes[i])
            
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
    
    #Turn off extra subplots if columns are fewer than grid spaces
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def violinplot_all(data: pd.DataFrame, target: str, log_scale: bool = False):
    """
    Plots violin plots for all numeric columns in a DataFrame, separated by the target class.
    Optional log scaling with log1p is applied to each numeric column if specified.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - target (str): The target column name, used to separate the violin plots by class.
    - log_scale (bool): If True, applies a log1p scale to each violin plot, default is False.
    """
    #Check if the target column exists in the DataFrame
    if target not in data.columns:
        raise ValueError(f"The target column '{target}' does not exist in the provided DataFrame.")
    
    #Ensure that the target column is categorical for use in the violin plot
    if data[target].dtype != 'object' and data[target].nunique() >= 10:
        warnings.warn(f"The target column '{target}' has a high number of unique values and may not be suitable for categorical plotting.")

    #Filter to select only the numerical columns for plotting
    numeric_cols = [col for col in data.columns if data[col].dtype != object and col != target]
    
    #Determine the number of rows and columns for the grid
    n_cols = 3  
    n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate rows based on columns and the number of plots
    
    #Set up the matplotlib figure with the calculated grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        # Apply log1p transformation if log_scale is True
        if log_scale:
            # Check for non-positive values in the data
            if (data[col] < 0).any():
                warnings.warn(f"Column '{col}' contains negative values and cannot use log scale. Plotting without log scale.")
                plot_data = data[col]  # Use original data if log scale isn't possible
                title = f"{col} by {target}"
                xlabel = col
            else:
                plot_data = np.log1p(data[col])  # Log1p-transform the data
                title = f'Log1p Transformed {col} by {target}'
                xlabel = f'Log1p of {col}'
        else:
            plot_data = data[col]
            title = f"{col} by {target}"
            xlabel = col

        #Plot violin plot, with target as the x-axis (hue is not needed for violin plots)
        sns.violinplot(data=data.assign(**{col: plot_data}), x=target, y=col, ax=axes[i])
            
        axes[i].set_title(title)
        axes[i].set_ylabel(xlabel)
    
    #Turn off extra subplots if columns are fewer than grid spaces
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


    