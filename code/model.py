import logging
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_model(args, maxlen, vocab):
    vocab_size = len(vocab)

    #####Input#####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape(args.neg_size,), dtype='int32', name='neg_input')

    #####Word embedding layer#####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    #####Sentence vector######
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    
