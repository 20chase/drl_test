import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def df_plot(dfs, x, ys, ylim=None, legend_loc='best'):
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.style.use('ggplot')
    if ylim:
        plt.ylim(ylim)

    plt.plot(dfs[x]/3600, dfs[ys], linewidth=1, label=ys)
    plt.xlabel(x)
    plt.legend(loc=legend_loc)
    plt.show()

def main():
    filepath='./ppo2/log/progress.csv'
    dataframes = []

    data = pd.read_csv(filepath)
    df_plot(data, 'time_elapsed', 'policy_entropy')
    df_plot(data, 'time_elapsed', 'eprewmean')
    df_plot(data, 'time_elapsed', 'policy_loss')
    df_plot(data, 'time_elapsed', 'explained_variance', ylim=(-1, 1))
    df_plot(data, 'time_elapsed', 'lr')
    df_plot(data, 'time_elapsed', 'value_loss')
    df_plot(data, 'time_elapsed', 'approxkl')

if __name__ == '__main__':
    main()
