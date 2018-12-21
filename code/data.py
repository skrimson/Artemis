import codecs
import re
import operator

def get_vocab(domain, maxlen=0, vocab_size=0):
    #maxlen: max length of sentences
    assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/'+domain+'/train.txt'

    total_words = 0
    word_freqs = {}

    source_txt = codecs.open(source, 'r', 'utf-8')
    for line in source_txt:
        words = line.split()
        #skip if sentence length is over maxlen
        if maxlen > 0 and len(words) > maxlen:
            continue

        for word in words:
            if not is_number(word):
                word_freqs[w] += 1
                total_words += 1

    print('{} total words'.format(total_words))
    sorted_words = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    #adding words to vocab
    for word, _ in sorted_words:
        vocab[words] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('using {} words most frequent'.format(vocab_size))

    #writing (word, frequence) to a file #no need
    vocab_file = codecs.open('../preprocessed_data/'+domain+'/vocab_freq', mode='w', encoding='utf-8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocabs:
        if index < 3:
            vocab_file.write(word+'\t'+str(0)+'\n')
            continue
        vocab_file.write(word+'\t'+str(word_freqs[word])+'\n')
    vocab_file.close()

    return vocab

def get_data(domain, phase, vocab, maxlen=0):
    assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    print('reading {} dataset'.format(domain))

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
