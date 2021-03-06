import argparse
import logging
import numpy as np
from time import time
import utils as U
from sklearn.metrics import classification_report
import codecs

#####Parsing Arguments#######
parser = argparse.ArgumentParser()
parser.add_argument("-ex", "--aspect-examples", dest="aspect_examples", type=str, metavar='<str>', nargs='*', required=True, help="aspect examples to be initialized")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
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
U.print_args(args)

from keras.preprocessing import sequence
import data as dataset

#######importing vocab###########
vocab, vocab_idf = dataset.get_vocab(args.domain, maxlen=args.maxlen, vocab_size=args.vocab_size)

#######Getting Test Data##########
test_x, test_max_len = dataset.get_data(args.domain, phase = 'test', vocab=vocab, maxlen=args.maxlen)
max_len = test_max_len

test_x = sequence.pad_sequences(test_x, maxlen=max_len)

######Building Model##############
from model import create_model
import keras.backend as K
from optimizers import get_optimizer

optimizer = get_optimizer(args)

def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)
model = create_model(args, max_len, vocab)

## Load the save model parameters
print("------loading model parameters------")
model.load_weights(out_dir+'/model_param')
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])

################ Evaluation ####################################

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []
    predict_out = codecs.open(out_dir + '/predicted_label.txt', 'w', 'utf-8')

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())
            predict_out.write(line.strip()+'\n')

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label,
            ['Food', 'Staff', 'Ambience', 'Price'], digits=3))

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            predict_label.append(label)
            predict_out.write(label+'\n')

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            true_label.append(label)

        print(classification_report(true_label, predict_label,
            ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:,-1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)

###### Creating a dictionary that map word index to word
print("-----creating dictionary------")
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input],
        [model.get_layer('att_weights').output, model.get_layer('p_t').output])
#weights for each word, weights for each aspect
att_weights, aspect_probs = test_fn([test_x])

###### Save attention weights on test sentences into a file
att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i!=0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(max_len-line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' '+str(round(weights[j], 3)) + '\n')

print("-----output finished------")

########F scores###########
cluster_map_r = {0: 'Food', 1: 'Staff', 2: 'Ambience', 3: 'Ambience'}
cluster_map_b = {0: 'feel', 1: 'taste', 2: 'smell', 3: 'look', 4: 'overall'}

print('--- Results on %s domain ---'.format(args.domain))
test_labels = '../preprocessed_data/' + args.domain + '/test_label.txt'
if args.domain == 'restaurant':
    prediction(test_labels, aspect_probs, cluster_map_r, domain=args.domain)
elif args.domain == 'beer':
    prediction(test_labels, aspect_probs, cluster_map_b, domain=args.domain)
