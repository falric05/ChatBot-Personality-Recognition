from matplotlib import pyplot as plt
from os.path import join
import numpy as np

def plot_datasets(characters_datasets: dict, 
                  colors:list[str] = ['red','orange','gold'],
                  out_plot_folder:str = None):
    """
    Produce a bar plot for all the datasets splitted in train, validation and test in one single graph.
    ## Params
    * `characters_datasets`: dictionary of pairs <Character, Dataset dictionarie>
    * `colors`: list of colors
    * `out_plot_folder`: where to store the plot
    """

    characters = list(characters_datasets.keys())
    characters_hg = list(characters_datasets.values())

    # Number of characters
    n = len(characters)
    # Number of sets for each character
    m = len(characters_hg[0])
    # List of datasets identyfier, contained in the dataset `characters_hg`
    sets = [s for s in characters_hg[0]]
    # The xlabels are the name of characters
    xlabels = characters
    # Extract lenght of each dataset and of each character, and store it in a vector
    data   = np.array([[len(characters_hg[c][s]) for c in range(n)] for s in sets])
    # Plots the dataset
    fig, ax = plt.subplots(facecolor='white', figsize=(15, 10))
    for s in range(m):
        ax.bar(xlabels, data[s], 0.3, color=colors[s])
    ax.set_xlabel('Set')
    ax.set_ylabel('Number of lines')
    ax.set_title('Barplot of Dataset of all characters', fontweight="bold")
    ax.legend([hgdict for hgdict in characters_hg[0]])
    ax.grid(True)
    # Save the figure
    if not out_plot_folder is None:
        fig.savefig(join(out_plot_folder, 'Datasets characters'), dpi=fig.dpi)
    return None