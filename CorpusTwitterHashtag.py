#move to here all corpus classes made for this dataset

from CorpusHelper import *

class TwitterHashtagCorpus(object):

    def __init__(self, files, vocab_file, dev_split=0.3, sent_max_length=50, vocab_size=8000):
        # loading data
        #files = [treino, validacao]
        self.dev_split = dev_split
        self.train_split = 0.7
        self.sent_max_length = sent_max_length
        self.vocab_size = vocab_size

        #treino
        with CorpusHelper.open_file(files[0]) as f:
            x_data_treino = []
            y_data_treino = []
            # skip header
            hashtags = f.readline()
            self.label_to_id = self.create_hashtags_file(files[0])#self.build_label_to_id(hashtags)
            self.max_labels_train = len(self.label_to_id)

            cont = 0
            for l in f:
                cont = cont + 1
                l = l.strip()
                ftrs = l.split('\t')
                text = ftrs[0]
                label = ftrs[-1]
                x_data_treino.append(text)
                ex_label = self.label_to_id[label.split()[0]]
                y_data_treino.append(ex_label) # Get just the first one hashtag

        # save vocabulary
        if not os.path.exists(vocab_file):
            CorpusHelper.build_vocab(x_data_treino, vocab_file, vocab_size)

        self.words, self.word_to_id = CorpusHelper.read_vocab(vocab_file)

        for i in range(len(x_data_treino)):  # tokenizing and padding
            x_data_treino[i] = CorpusHelper.process_text(x_data_treino[i], self.word_to_id, sent_max_length, clean=False)

        # print(x_data)
        x_data_treino = np.array(x_data_treino)
        y_data_treino = np.array(y_data_treino)


        with CorpusHelper.open_file(files[1]) as f:
            x_data_val = []
            y_data_val = []
            # skip header
            hashtags = f.readline()
            self.label_to_id = self.create_hashtags_file(files[1])#self.build_label_to_id(hashtags)
            self.max_labels = len(self.label_to_id)

            for l in f:
                l = l.strip()
                ftrs = l.split('\t')
                text = ftrs[0]
                label = ftrs[-1]
                x_data_val.append(text)
                ex_label = self.label_to_id[label.split()[0]]
                y_data_val.append(ex_label) # Get just the first one hashtag

        # save vocabulary
        if not os.path.exists(vocab_file):
            CorpusHelper.build_vocab(x_data_val, vocab_file, vocab_size)

        self.words, self.word_to_id = CorpusHelper.read_vocab(vocab_file)

        for i in range(len(x_data_val)):  # tokenizing and padding
            x_data_val[i] = CorpusHelper.process_text(x_data_val[i], self.word_to_id, sent_max_length, clean=False)

        # print(x_data)
        x_data_val = np.array(x_data_val)
        y_data_val = np.array(y_data_val)


        self.x_train = x_data_treino
        self.y_train = y_data_treino

        self.x_validation = x_data_val
        self.y_validation = y_data_val

        self.x_test = []
        self.y_test = []


    def __str__(self):
        return 'Training: {},Validation{},Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_validation),
                                                                               len(self.x_test), len(self.words))

    def shuffle(self, dev=None):
        if dev is None:
            dev = self.dev_split
        x = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test), axis=0)
        indices = np.random.permutation(np.arange(len(x)))
        x_data = x[indices]
        y_data = y[indices]

        # train/dev split
        dtsize = len(x_data)
        num_train = int(self.train_split * dtsize)
        num_test = int(dtsize * self.dev_split)
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:num_train + num_test]
        self.y_test = y_data[num_train:num_train + num_test]
        self.x_validation = x_data[num_train + num_test:]
        self.y_validation = y_data[num_train + num_test:]

    def build_label_to_id(self, hashtags):
        hashtags = re.sub("'",'', hashtags[1:-1])
        hashtags = re.split(r", ", hashtags)
        labels_dict = dict(zip(hashtags, range(len(hashtags))))
        return labels_dict

    def create_hashtags_file(self, train_file, label_file='twitter_hashtag/hashtags_clean.label'):
        id = 0
        with CorpusHelper.open_file(train_file, 'r') as inFile:
            inFile.readline()
            for l in inFile:
                labels = l.split('\t')[-1].split()
               # print(labels,"\n\n")
                for label in labels:
                    if id == 0:
                        label_dict = {label: id}
                        id = id + 1
                    else:
                        if label not in label_dict:
                            dt = {label: id}
                            id = id + 1
                            label_dict.update(dt)

        with CorpusHelper.open_file(label_file, 'w') as out_file:
            hashtags = label_dict.keys()
            for hashtag in hashtags:
                out_file.write(hashtag + '\n')

        #print(len(label_dict))
        return label_dict

#twitter_corpus = TwitterHashtagCorpus('out.txt','twitterhashtags.vocab')
#print(twitter_corpus)
