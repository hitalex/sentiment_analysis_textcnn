#coding=utf8

import numpy as np
import re
import word2vec
# import itertools
# from collections import Counter
# import codecs

class w2v_wrapper:
     def __init__(self,file_path):
        # w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.model = word2vec.load(file_path)
        if 'unknown' not  in self.model.vocab_hash:
            unknown_vec = np.random.uniform(-0.1,0.1,size=128)
            self.model.vocab_hash['unknown'] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors,unknown_vec))

# 读入下载的词向量模型
class WordEmbeddingModel:
    def __init__(self, file_path):
        with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
            line = f.readline()
            seg_list = line.split(' ')
            self.vocab_size = int(seg_list[0])
            self.embedding_dim = int(seg_list[1])

            print('Vocabulary size: ', vocab_size, '\tVector dim: ', embedding_dim)

            self.vocab_hash = dict()
            self.vectors = np.zeros((vocab_size, embedding_dim), np.float)
            i = 0
            for line in f:
                seg_list = line.split(' ')
                word = seg_list[0]
                assert embedding_dim == len(seg_list[1:])
                self.vectors[i] = list(map(float, seg_list[1:]))
                self.vocab_hash[word] = self.vectors[i]
                i += 1

        # 添加未出现的词
        if 'unknown' not  in self.vocab_hash:
            unknown_vec = np.random.uniform(-0.1,0.1, size = embedding_dim)
            self.vocab_hash['unknown'] = len(self.vocab_hash)
            self.vectors = np.row_stack((self.vectors,unknown_vec))

        self.vocab_size = len(self.vocab_hash)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def removezero( x, y):
    nozero = np.nonzero(y)
    print('removezero',np.shape(nozero)[-1],len(y))

    if(np.shape(nozero)[-1] == len(y)):
        return np.array(x),np.array(y)

    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x, y


def read_file_lines(filename,from_size,line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if(i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text



def load_data_and_labels(filepath,max_size = -1):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_datas = []

    with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
        train_datas = f.readlines()

    one_hot_labels = []
    x_datas = []
    for line in train_datas:
        parts = line.split('\t',1)
        if(len(parts[1].strip()) == 0):
            continue

        x_datas.append(parts[1])
        if parts[0].startswith('0') :
            one_hot_labels.append([0,1])
        else:
            one_hot_labels.append([1,0])

    print (' data size = ' ,len(train_datas))

    # Split by words
    # x_text = [clean_str(sent) for sent in x_text]

    return [x_datas, np.array(one_hot_labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
            yield shuffled_data[start_index:end_index]


def get_text_idx(text,vocab,max_document_length):
    ''' 从已训练好的词向量模型初始化
    Input:
        text: 输入的文本
        vocab：预处理的词向量模型map
        max_document_length: 最长句子中词的数量
    Output:
        词向量的矩阵
    '''
    text_array = np.zeros([len(text), max_document_length],dtype=np.int32)

    for i,x in  enumerate(text):
        words = x.split(" ")
        for j, w in enumerate(words):
            if w in vocab:
                text_array[i, j] = vocab[w]
            else :
                text_array[i, j] = vocab['unknown']

    return text_array


if __name__ == "__main__":
    x_text, y = load_data_and_labels('F:\BaiduYunDownload\SentimentAnalysis\corpus_ch\cutclean_stopword_corpus10000.txt')
    print (len(x_text))
