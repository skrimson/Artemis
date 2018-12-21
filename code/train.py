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

####optimizer algorithm####
from optimizers import get_optimizer
optimizer = get_optimizer(args)

####Building Model########################################
from model import create_model
import keras.backend as K

logger.info('  Building model')

def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_model(args, overall_maxlen, vocab, vocab_idf)
# freeze the word embedding layer, because using pre-trained word embeddings
model.get_layer('word_emb').trainable=False
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
