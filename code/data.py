import codecs
import re
import operator
import numpy as np
import math

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))


def get_vocab(domain, maxlen=0, vocab_size=0):
    #maxlen: max length of sentences
    assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/'+domain+'/train.txt'

    total_words = 0
    word_freqs = {}
    document_count = 0

    #counting word frequency and word in document frequency
    source_txt = codecs.open(source, 'r', 'utf-8')
    document_freq = {}
    for line in source_txt:
        words = line.split()
        words_doc = list(set(words))
        #skip if sentence length is over maxlen
        if maxlen > 0 and len(words) > maxlen:
            continue

        #counting words
        for word in words:
            if not is_number(word):
                #trying if it has 0 count or not
                try:
                    word_freqs[word] += 1
                except KeyError:
                    word_freqs[word] = 1
                total_words += 1

        #IDF: counting word in doc
        for word in words_doc:
            if not is_number(word):
                #trying if it has 0 count or not
                try:
                    document_freq[word] += 1
                except KeyError:
                    document_freq[word] = 1
        document_count += 1
    document_freq['<pad>'] = 0
    document_freq['<unk>'] = 0
    document_freq['<num>'] = 0

    print('{} total words'.format(total_words))
    sorted_words = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)

    #adding words to vocab
    for word, _ in sorted_words:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('using {} words most frequent'.format(vocab_size))

    #writing (word, frequence) to a file
    vocab_file = codecs.open('../preprocessed_data/'+domain+'/vocab_freq', mode='w', encoding='utf-8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word+'\t'+str(0)+'\n')
            continue
        vocab_file.write(word+'\t'+str(word_freqs[word])+'\n')
    vocab_file.close()

    #IDF: computing and writing (word_id, idf) to a file
    idf_file = codecs.open('../preprocessed_data/'+domain+'/vocab_idf', mode='w', encoding='utf-8')
    vocab_idf = {}
    for word, index in sorted_vocab:
        if index < 3:
            idf_file.write(str(index)+'\t'+str(0)+'\n')
            continue
        vocab_idf[index] = math.log(document_count / float(document_freq[word]))
        idf_file.write(str(index)+'\t'+str(vocab_idf[index])+'\n')
    idf_file.close()

    return vocab, vocab_idf

def get_data(domain, phase, vocab, maxlen=0):
    assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    print('reading {} {} dataset'.format(domain, phase))

    source = '../preprocessed_data/'+domain+'/'+phase+'.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    source_txt = codecs.open(source, 'r', 'utf-8')
    for line in source_txt:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    return data_x, maxlen_x
