
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
    plt.figure(figsize=(6, max(5, len(missing_count) * 0.3)))  # Adjust height based on number of features
    plt.title(f'Missing values over the {len(usable)} usable samples')
    
    plt.barh(np.arange(len(missing_count)), 
             missing_count['null_ratio'], 
             color='red', label='missing')
    
    plt.barh(np.arange(len(missing_count)), 
             1 - missing_count['null_ratio'],
             left=missing_count['null_ratio'],
             color='gray', label='available')
    
    plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlim(0, 1)
    plt.xlabel("Percentage")
    plt.legend()
    plt.tight_layout()
    plt.show()





#without the exception handling

"""def missingvalues_plot(data: pd.DataFrame, target: str):

    usable = data[data['target'].notnull()]

    missing_count = usable.isnull().sum().reset_index()
    missing_count.columns = ['feature', 'null_count']
    missing_count['null_ratio'] = missing_count['null_count'] / len(usable)

    missing_count = missing_count.sort_values('null_count', ascending=False)

    plt.figure(figsize=(6, 15))
    plt.title(f'Missing values over the {len(usable)} usable samples')

    plt.barh(np.arange(len(missing_count)), 
            missing_count['null_ratio'], 
            color='red', label='missing')

    plt.barh(np.arange(len(missing_count)), 
            1 - missing_count['null_ratio'],
            left=missing_count['null_ratio'],
            color='gray', label='available')

    plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlim(0, 1)
    plt.legend()
    plt.show()"""

    