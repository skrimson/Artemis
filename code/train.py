import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

####parsing arguments####
parser = argparse.ArgumentParser()
parser.add_argument("-ex", "--aspect-examples", dest="aspect_examples", type=str, metavar='<str>', nargs='*', required=True, help="aspect examples to be initialized")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="Path to the output")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant', help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
U.mkdir_p(out_dir)
U.print_args(args)

####Preparing Data########################################

from keras.preprocessing import sequence
import data as dataset

#importing vocabs
vocab, vocab_idf = dataset.get_vocab(args.domain, maxlen=args.maxlen, vocab_size=args.vocab_size)
#vocabulary with noun only

#train, test data
train_x, train_max_len = dataset.get_data(args.domain, phase = 'train', vocab=vocab, maxlen=args.maxlen)
test_x, test_max_len = dataset.get_data(args.domain, phase = 'test', vocab=vocab, maxlen=args.maxlen)
overall_maxlen = max(train_max_len, test_max_len)

train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

print('{} training examples'.format(len(train_x)))
print('Length of vocab: {}'.format(len(vocab)))

def sentence_batch_generator(data, batch_size):
    n = len(data) / batch_size
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1
        yield batch

def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        neg_samples = data[indices].reshape(batch_size, neg_size, dim)
        yield neg_samples

####Optimizer Algorithm###################################
from optimizers import get_optimizer
optimizer = get_optimizer(args)

####Building Model########################################
from model import create_model
import keras.backend as K

logger.info('  Building model with {} as aspect initialization'.format(args.aspect_examples))

def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_model(args, overall_maxlen, vocab)
# freeze the word embedding layer, because using pre-trained word embeddings
model.get_layer('word_emb').trainable=False
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])

#####Training#############################################
from keras.models import load_model
from tqdm import tqdm

logger.info('--------------------------------------------------------------------------------------------------------------------------')

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

batches_per_epoch = 1000
min_loss = float('inf')

for epoch in range(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    sentence_generator = sentence_batch_generator(train_x, args.batch_size)
    negative_generator = negative_batch_generator(train_x, args.batch_size, args.neg_size)

    for batch in tqdm(range(batches_per_epoch)):
        sen_input = sentence_generator.__next__()
        neg_input = negative_generator.__next__()

        batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input], np.ones((args.batch_size, 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    tr_time = time() - t0

    if loss < min_loss:
        min_loss = loss
        word_emb = np.asarray(model.get_layer('word_emb').get_weights())
        word_emb = word_emb.reshape(word_emb.shape[1], word_emb.shape[2])
        aspect_emb = np.asarray(model.get_layer('aspect_emb').get_weights())
        aspect_emb = aspect_emb.reshape(aspect_emb.shape[1], aspect_emb.shape[2])
        word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
        aspect_emb = aspect_emb / np.linalg.norm(aspect_emb, axis=-1, keepdims=True)
        aspect_file = codecs.open(out_dir+'/aspect.log', 'w', 'utf-8')
        model.save_weights(out_dir+'/model_param')

        #arranging words similar to aspect
        for i in range(len(aspect_emb)):
            aspect = aspect_emb[i]
            similarity = word_emb.dot(aspect.T)
            ordered_words = np.argsort(similarity)[::-1]
            word_list = [vocab_inv[w] for w in ordered_words[:300]]
            print('Aspect ({}):'.format(args.aspect_examples[i]))
            print(word_list)
            aspect_file.write('Aspect ({}):\n'.format(args.aspect_examples[i]))
            aspect_file.write(' '.join(word_list) + '\n\n')

    logger.info('Epoch %d, train: %is' % (epoch, tr_time))
    logger.info('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (loss, max_margin_loss, loss-max_margin_loss))

    sentence_generator.close()
    negative_generator.close()
