import logging
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_model(args, maxlen, vocab):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(int(w_n.shape[0]))))
        return args.ortho_reg*reg

    vocab_size = len(vocab)
    aspect_size = len(args.aspect_examples)

    #####Input#####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size,maxlen), dtype='int32', name='neg_input')

    #####Word Embedding#####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    #####Sentence Vector######
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    #d_i and M are included in this layer
    a_w = Attention(name='att_weights')([e_w, y_s])
    z_s = WeightedSum()([e_w, a_w])

    ##### Compute representations of negative instances #####
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction #####
    p_t = Dense(aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(aspect_size, args.emb_dim, name='aspect_emb', W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(inputs=[sentence_input, neg_input], outputs=loss)
    model.summary()

    ### Word embedding and aspect embedding initialization ######
    if args.emb_path:
        from emb_reader import EmbeddingReader as EmbeddingReader
        emb_reader = EmbeddingReader(args.emb_path, emb_dim=args.emb_dim)
        logger.info('Initializing word embedding matrix')
        model.get_layer('word_emb').set_weights(emb_reader.get_word_emb(vocab, model.get_layer('word_emb').get_weights())[np.newaxis,:,:])
        logger.info('Initializing aspect embedding from {}'.format(args.aspect_examples))
        model.get_layer('aspect_emb').set_weights(emb_reader.get_aspect_emb(args.aspect_examples)[np.newaxis,:,:])
    return model
