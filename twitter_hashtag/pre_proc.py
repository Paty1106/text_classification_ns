import re
import numpy as np
import random
import csv
import pandas as pd
import json

# primeira linha ['#afas', ... ,'#fasf']
# frase ##HASHTAG## tab #umrei

def pre_proc():

    f = open('twitter_hashtag/17mi-dataset_preprocessed.txt', mode='r')
    w = open('twitter_hashtag/17mi-dataset_clean.txt', mode='w+', encoding="utf-8")
    c = csv.writer(open("hash.csv", "wb"))

    hashtags = []
    cont = 0
    hash = 0
    for l in f:
        print(cont)
        if cont == 200000:
            d = {'h': hashtags}
            with open('twitter_hashtag/hashs.json', 'w') as outfile:
                json.dump(d, outfile, indent=4)

            f.close()
            w.close()
            break

        l_split = l.split('\t')[1]
        words = l_split.split(' ')

        w_hash = []
        for word in words:
            if word.find('#') == 0:
                hash = 1
                w_hash.append(word.rstrip("\n"))
                l_split = l_split.replace(word," ##HASHTAG## ")
               # word = word.rstrip("\n")

        if hash == 1:
            i = random.randint(0, len(w_hash)-1)

            line = '{}\t{}\n'.format(l_split.rstrip(), w_hash[i].strip())
            print(line)
            w.write(line)

            if hashtags.count(w_hash[i]) == 0:
                hashtags.append(w_hash[i])

            cont+=1
            hash = 0

def hash():
     f = open('twitter_hashtag/17mi-dataset_preprocessed.txt', mode='r')



pre_proc()
