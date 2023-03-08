from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def barplot(metrics: dict, title: str,
            logscale=True, figsize=(16,12), colormap='viridis'):
    """
    Plots a barplot starting from a dictionary dict
    ## Params
    * `metrics`: dictionary to used to create the correlation matrix
    * `title`: title of plot
    * `logscale`: if `True` the barplot is plotted with symlog on y axis, default `True`
    * `figsize`: dimension of plot
    * `colormap`: colormap selected
    """
    # create the dataframe from the dictionary
    df = pd.DataFrame(data=metrics)
    # plot and take the axes
    ax = df.plot(kind='bar', x='metrics',  
                 colormap=colormap, figsize=figsize)
    # in case you need logscale on y axis, it is setted
    if logscale:
        ax.set_yscale('symlog')
    # show the plot
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel('score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid()
    plt.show()

def corrplot(metrics: dict, inverted: bool, title: str, debug=False):
    """
    Plots a correlation matrix starting from the dictionary `metrics`
    ## Params
    * `metrics`: dictionary to used to create the correlation matrix
    * `inverted`: if `True` the dictionary is transposed
    * `title`: title of plot
    * `debug`: if `True` the dictionary can be printed, default `False`
    """
    # debug
    if debug: print(metrics)
    # create the dataframe from the dictionary
    df = pd.DataFrame(data=metrics)
    if debug: print(df)
    # in case you need to invert the dictionary
    if inverted:
        dfT = df.T
        #set column names equal to values in row index position 0
        dfT.columns = dfT.loc['metrics']
        #remove first row from DataFrame
        dfT = dfT.drop('metrics')
        #reset index
        dfT.reset_index(inplace=True, names=['characters']+df['metrics'].to_list())
        #change dtypes
        for c in dfT.columns[1:]:
            dfT[c] = pd.to_numeric(dfT[c])
        dfT = dfT.drop('characters', axis=1)
        df = dfT
    else:
        # otherwise
        df = df.drop('metrics', axis=1)
    # metrics
    if debug: print(df)
    # compute the correlation metrix
    corr = df.corr()
    # and plots it
    sns.heatmap(corr, annot = True)
    plt.title(title)
    plt.show()

def corrm(corr: any, title: str):
    """Plots a correlation matrix with its title"""
    plt.figure(figsize=(18,16))
    sns.heatmap(corr, annot = True)
    plt.title(title)
    plt.show()