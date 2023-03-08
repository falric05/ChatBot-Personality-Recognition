# reference to https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        data = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def example_data():
    # The following data is from the Barney Stinson character of How I Met Your Mother.
    #
    # The data contains 6 kinds of emotions a character could have. First element of data
    # contains the emotion labels used for evaluating character emotions.
    # Predictions-Label is a couple of list of the resulting emotion values of the evaluation
    # for predictions given by the chatbot and resulting emotion values of the evaluation for
    # labels
    data = [
        ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
        ('Predictions-Labels', [
            [0.04045128605018059,
             0.3372649804999431,
             0.029534351934368413,
             0.33119267721970874,
             0.24915542546659708,
             0.012401290975200633],
            [0.04045128605018059,
             0.2372649804999431,
             0.029534351934368413,
             0.23119267721970874,
             0.24915542546659708,
             0.112401290975200633]])
    ]
    return data

class EmotionsRadar:
    data = []
    character = ''

    def __init__(self, emotions, predictions, labels, character):
        self.data = [
            emotions,
            ('Predictions-Labels', [predictions, labels])
        ]
        self.character = character
        
    def subplotEmotionsRadar(self, nrows, ncols, fig, idx, legend=None, colors=None):
        # Plots a set of `nrows` rows and `ncols` columns subplots with index idx
        N = len(self.data[0])
        theta = radar_factory(N, frame='polygon')
        spoke_labels = self.data.pop(0)
        axs = fig.add_subplot(nrows, ncols, idx, projection='radar')
        axs.set_title('Emotion ' + self.character, weight='bold', size='medium', position=(0.5, 1.1),
                      horizontalalignment='center', verticalalignment='center')
        if colors == None:
            colors = ['b', 'r']
        # Plot the four cases from the example data on separate axes
        for (title, case_data) in self.data:
            axs.set_rgrids([0.2, 0.4, 0.6, 0.8])
            for d, color in zip(case_data, colors):
                axs.plot(theta, d, color=color, label='_nolegend_')
                axs.fill(theta, d, facecolor=color, alpha=0.25)
            axs.set_varlabels(spoke_labels)
        # add legend relative to top-left plot
        if legend==None:
            legend = ('Predictions', 'Labels')
        legend = axs.legend(legend, loc=(0.9, .95),
                            labelspacing=0.1, fontsize='small')

    def plotEmotionsRadar(self, titleplot: str):
        # Plots a radar plot 
        N = len(self.data[0])
        theta = radar_factory(N, frame='polygon')
        spoke_labels = self.data.pop(0)
        fig, axs = plt.subplots(figsize=(N, N), subplot_kw=dict(projection='radar'))
        colors = ['b', 'r']
        # Plot the four cases from the example data on separate axes
        for (title, case_data) in self.data:
            axs.set_rgrids([0.2, 0.4, 0.6, 0.8])
            axs.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                          horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, colors):
                axs.plot(theta, d, color=color, label='_nolegend_')
                axs.fill(theta, d, facecolor=color, alpha=0.25)
            axs.set_varlabels(spoke_labels)
        # add legend relative to top-left plot
        labels = ('Predictions', 'Labels')
        legend = axs.legend(labels, loc=(0.9, .95),
                            labelspacing=0.1, fontsize='small')
        fig.text(0.5, 0.965, titleplot + ' ' + self.character,
                 horizontalalignment='center', color='black', weight='bold',
                 size='large')
        plt.show()