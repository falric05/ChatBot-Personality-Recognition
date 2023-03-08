from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def plot_wordcloud(lines:str, cmap='viridis', title=None):
    """
    Plots a WordCloud of a list of lines of a given tv show character
    ## Params 
    * `lines`: list of lines of a given tv show character
    * `cmap`: colormap for words in the wordcloud
    * `title`: title of plot
    """
    # initialize the wordcloud setting the plot parameters
    wordcloud = WordCloud(background_color = 'black', width = 800, height = 400,
                          colormap = cmap, max_words = 100, contour_width = 3,
                          max_font_size = 80, contour_color = 'steelblue', 
                          stopwords = set(STOPWORDS), random_state = 0)
    # generate the wordcloud from text
    wordcloud.generate(lines.lower())
    # if title is not None
    if not (title is None):
        plt.title(title)
    # show the plot
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figure()