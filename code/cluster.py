from gensim.models import Word2Vec
import codecs

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class MySentences():
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()

def display_closestwords_tsnescatterplot(model, word):

    arr = np.empty((0,200), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

source = '../preprocessed_data/restaurant/train.txt'
sentences = MySentences(source)
model = Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)

display_closestwords_tsnescatterplot(model, 'food')

# X = []
# for word in model.wv.vocab:
#     X.append(model.wv[word])
#
# X = np.array(X)
# print("Computed X: ", X.shape)
# X_embedded = TSNE(n_components=2, n_iter=250, verbose=2).fit_transform(X)
# print("Computed t-SNE", X_embedded.shape)
#
# df = pd.DataFrame(columns=['x', 'y', 'word'])
# df['x'], df['y'], df['word'] = X_embedded[:,0], X_embedded[:,1], model.wv.vocab
#
# source = ColumnDataSource(ColumnDataSource.from_df(df))
# labels = LabelSet(x="x", y="y", text="word", y_offset=8, text_font_size="8pt", text_color="#555555", source=source, text_align='center')
#
# plot = figure(plot_width=600, plot_height=600)
# plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
# plot.add_layout(labels)
# show(plot, notebook_handle=True)
