import gensim
import codecs

class MySentences():
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()

def word2vec(domain):
    source = '../preprocessed_data/' + domain + '/train.txt'
    model_file = '../preprocessed_data/' + domain + '/w2v_embedding'
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)

if __name__ == '__main__':
    print('------pre-training word embeddings------')
    word2vec('restaurant')
    word2vec('beer')
