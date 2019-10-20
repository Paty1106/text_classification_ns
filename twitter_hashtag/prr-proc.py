import re
import numpy as np

def test_mul():

    f = open('./f.txt', mode='r')
    w = open('./multiple.txt', mode='w+', encoding="utf-8")

    fline = f.readline()
    w.write(fline)

    """hash = fline.split(', ')
    index = [x for x in range(125)]
    hashtags = [str(h[1:-1]) for h in hash]
    hashtags[0] = hashtags[0][1:]
    hashtags[-1] = hashtags[-1][:-2]
    print(hashtags)
    c_dict = dict(zip(hashtags, index))"""
    counter = np.zeros(125)
    c_dict = dict()
    index = 0
    for l in f:
        t_h = l.split('\t')
        print(t_h)

        h = t_h[1].split(' ')
        for hashtag in h:
            if hashtag[-1]=='\n':
                h_mod = hashtag[:-1]
            else:
                h_mod = hashtag
            try:
                if (counter[c_dict[h_mod]] >= 1000):
                    continue
                line = '{}\t{}\n'.format(t_h[0], h_mod)
                print(line)
                w.write(line)
                counter[c_dict[h_mod]] += 1
            except :
                pair = {h_mod: index}
                index += 1
                c_dict.update(pair)
                print(c_dict)
                line = '{}\t{}\n'.format(t_h[0], h_mod)
                print(line)
                w.write(line)
                counter[c_dict[h_mod]] += 1





#multiplicate_exemples()
test_mul()