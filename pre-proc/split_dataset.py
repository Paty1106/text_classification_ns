import numpy as np
import pandas as p
import csv
train_file_path ="twitter_hashtag.train"
test_file_path = "twitter_hashtag.test"


def split_dataset(file, split):
    train_file = open(train_file_path, 'w', encoding='utf-8', errors='ignore')
    test_file = open(test_file_path, 'w', encoding='utf-8', errors='ignore')

    data = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        hashtags = f.readline()
        for l in f:
            data.append(l)

        data_len = len(data)
        sdata = np.array(data)
        index = np.random.permutation(np.arange(data_len))
        sdata = data[index]
        split_index = split*data_len
        train_data = sdata[:split_index]
        test_data = sdata[split_index:]

def split_data(file):
    df = p.read_csv(file, skiprows=1, sep='\n')
    print(df.sample(10))
    print(len(df))
    train = df.sample(frac=0.8, random_state=200)
    print(len(train.index))
    test = df.drop(train.index)
    print(len(train))
    print(len(test))
    train.to_csv(train_file_path, index=None)
    test.to_csv(test_file_path, index=None)

#split_dataset("out.txt", 0.8)
split_data("out.txt")