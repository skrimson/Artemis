import codecs
import logging
import numpy as np
import gensim
from sklearn.cluster import KMeans
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingReader:
    def __init__(self, emb_path, emb_dim=None):

        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        model = gensim.models.Word2Vec.load(emb_path)
        self.emb_dim = emb_dim

        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        if emb_dim != None:
            assert self.emb_dim == len(self.embeddings['nice'])

        self.vocab_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))

    def get_word_emb(self, vocab, emb_matrix):
        counter = 0.
        embs = np.asarray(emb_matrix).reshape(len(vocab), self.emb_dim)
        for word, index in vocab.items():
            try:
                embs[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info('{} words initilialized'.format(counter))
        # L2 normalization
        normalized_embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
        return normalized_embs

    def get_aspect_emb(self, aspect_words):
        aspect_embs = []
        for word in aspect_words:
            aspect_embs.append(self.embeddings[word])
        aspect_embs = np.asarray(aspect_embs)
        normalized_aspect_embs =  aspect_embs / np.linalg.norm(aspect_embs, axis=-1, keepdims=True)

        return normalized_aspect_embs
