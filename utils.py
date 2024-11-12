
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter

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
    # Filter usable samples where target is not null and include only numeric columns
    #usable = numeric_data[numeric_data[target].notnull()]


    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("Invalid correlation method. Choose from 'pearson', 'spearman', or 'kendall'.")


    if usable.empty:
        print("No usable samples found (all target values are missing).")
        return
    
    # Calculate the correlation matrix

    corr = usable.corr(method = method)
    corr = corr.dropna(how='all').dropna(axis=1, how='all')

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(max(0.5 * num_cols, 10), max(0.5 * num_cols, 10)))
    plt.title(f'Correlation matrix over the {len(usable)} usable samples with {method} correlation')

    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.color_palette("rocket", as_cmap=True), center=0)
    plt.tight_layout()
    plt.show()
    

def histplot_all(data: pd.DataFrame, target: str):

    for col in data.columns:
        if col.dtype != object and col != target:
            sns.histplot(data[col], kde=True)
            plt.title(f'{col} distribution')
            plt.show()

            

            