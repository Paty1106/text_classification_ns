from CorpusHelper import CorpusHelper


def createhashtags_file(label_file, train_file):
    id = 0
    with CorpusHelper.open_file(train_file,'r') as inFile:
        inFile.readline()
        for l in inFile:
            labels = l.split('\t')[-1].split()
            for label in labels:
                if id == 0:
                    label_dict = {label: id}
                    id = id +1
                else:
                    if label not in label_dict:
                        dt = {label: id}
                        id = id + 1
                        label_dict.update(dt)


    with CorpusHelper.open_file(label_file,'w') as out_file:
        hashtags = label_dict.keys()
        for hashtag in hashtags:
            out_file.write(hashtag+'\n')


createhashtags_file('hashtags.label', 'out.txt')