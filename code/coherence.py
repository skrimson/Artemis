from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np

def get_coherence(aspect_num):
    #######aspects as topics#########
    aspect_txt = open("./output_dir/" + "restaurant" + "/aspect.log")
    aspect_words = []
    for line in aspect_txt:
        words = line.split()
        if len(words) <= 5 or "Aspect" in words[0]:
            continue
        aspect_words.append([words[i] for i in range(aspect_num) if not (words[i] == "<unk>") or (words[i] == "<num>") or (words[i] == "<pad>")])

    #######creating model###########
    cm = CoherenceModel(topics=aspect_words, corpus=corpus, dictionary=dic, coherence='u_mass')

    ######compute###########
    coherence = cm.get_coherence()

    return coherence

#######creating dictionary, corpus from test data#########
test_txt = "../preprocessed_data/" + "restaurant" + "/train.txt"
sentences = [s for s in word2vec.LineSentence(test_txt)]
dic = Dictionary(sentences)
corpus = [dic.doc2bow(s) for s in sentences]

#######creating graph#########
x = []
y = []
for i in range(1, 50, 1):
    score = get_coherence(i)
    x.append(i)
    y.append(score)
    print("{} number of aspects: {}".format(i,score))

plt.xlabel('Top N Words')
plt.ylabel('Coherence Score')
plt.plot(x, y)
plt.show()
