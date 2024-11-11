
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter

def missingvalues_plot(data = pd.DataFrame, target = str):

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
    plt.show()