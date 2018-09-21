#coding=utf8

import numpy as np
import re
import word2vec
# import itertools

# from collections import Counter

# import codecs

subject_list = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']
sentiment_list = [-1, 0, 1]

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
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
            line = f.readline()
            seg_list = line.split(' ')
            vocab_size = int(seg_list[0])
            embedding_dim = int(seg_list[1])
            print('Vocabulary size: ', vocab_size, '\tVector dim: ', embedding_dim)

            #import ipdb; ipdb.set_trace()
            self.vocab_hash = dict()
            self.vectors = np.zeros((vocab_size, embedding_dim), np.float)
            i = 0
            for line in f:
                seg_list = line.strip().split(' ')
                word = seg_list[0].strip()
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
        self.embedding_dim = embedding_dim


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


def load_text_subject_sentiment_labels(file_path):
    """
    导入文本并进行文本预处理，以及subject和sentiment值
    Input:
        file_path: 训练文件的路径
    Output:
        content_id_list, text_feature_list, subject_labels, sentiment_labels
    """
    import pandas as pd
    import numpy as np

    data = pd.read_csv(file_path)
    content_id_list = list(data['content_id'])

    subject_id_map = dict()
    for i, s in enumerate(subject_list):
        subject_id_map[s] = i

    total = len(content_id_list)
    text_feature_list = [0] * total
    sentiment_labels = np.zeros((total, 3))
    subject_labels = np.zeros((total, len(subject_list)))
    content_id_map = dict() # key: content_id, value: 已经预处理好的文本
    for i, content in enumerate(data['content']):
        content_id = data['content_id'][i]
        if content_id in content_id_map:
            # 因为有可能一条信息有多个subject的标注，而对应的content是一样的
            text_feature_list[i] = content_id_map[content_id]
        else:
            text_feature_list[i] = preprocess_text(content)

        assert data['subject'][i] in subject_id_map and data['sentiment_value'][i] in [-1, 0, 1]
        subject_index = subject_id_map[data['subject'][i]]
        subject_labels[i, subject_index] = 1
        sentiment_labels[i, data['sentiment_value']] = 1  # [中立，正向，负向]

    return content_id_list, text_feature_list, subject_labels, sentiment_labels


def preprocess_text(text):
    """ 对文本进行预处理，包括分词、去除停用词
    Input:
        text: 文本内容
    Output:
        outtext
    """
    import jieba
    text = trim_text(text) # 去除空白符号
    text = strQ2B(text) # 去除半角符号
    stopwords = load_stopwords("../stopword.txt")
    seg_list = list(jieba.cut(text))
    new_seg_list = []
    for w in seg_list:
        if w not in stopwords:
            new_seg_list.append(w)

    return new_seg_list


def trim_text(text):
    """ 删除文本中的所有空白符号
    """
    import re
    text = text.replace(u'\xa0','') # 删除&nbsp;
    text = re.sub('\s+', ' ', text, re.UNICODE) # 将连续的空白符替换成单个空白符
    text = text.strip()
    return text


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        #rstring += unichr(inside_code)
        rstring += chr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        #rstring += unichr(inside_code)
        rstring += chr(inside_code)
    return rstring


def load_stopwords(path):
    with open(path, 'r', encoding='utf-8',errors='ignore') as f:
        stopwords = []
        for line in f:
            line = line.strip()
            stopwords.append(line)

        return stopwords

    return None


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
        #words = x.split(" ")
        words = x
        for j, w in enumerate(words):
            if w in vocab:
                text_array[i, j] = vocab[w]
            else:
                text_array[i, j] = vocab['unknown']

    return text_array


if __name__ == "__main__":
    x_text, y = load_data_and_labels('F:\BaiduYunDownload\SentimentAnalysis\corpus_ch\cutclean_stopword_corpus10000.txt')
    print(len(x_text))
