import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
import time

class basic_lda():
    def __init__(self,alpha = 5,beta = 5,topic_num= 5):
        self.beta = beta
        self.alpha = alpha
        self.k = topic_num  # 话题数量
        self.loss_l = []

        #self.initlize()

    def read_dat(self,path):
        print ('starting')
        self.path = path
        self.data = np.load(self.path)
        print ('Finish')

    def initize(self):
        print ('Starting Initization')
        self.n = max(self.data[:, 1]) + 1  # 文芳数量
        self.words_num = max(self.data[:, 0]) + 1  # 单词数量
        print (self.n,self.words_num)
        self.doc_word_sum = np.zeros(self.n)  # 每个文档单词总数 n×1
        self.topic_word_sum = np.zeros(self.k)  # 每个主题的单词总数
        self.doc_topic_count = np.zeros((self.n, self.k))  # doc-topic count
        self.word_topic_count = np.zeros((self.words_num, self.k))  # word-topic count
        self.topic_att = []  # 每个单词的主题
        print (self.doc_word_sum.shape,self.topic_word_sum.shape, self.doc_topic_count.shape)

        for  word_idx,doc_idx, word_num in self.data:
            topic = np.random.randint(0, self.k, 1)[0]  # 随机初始化主题
            self.topic_att.append(topic)

            self.doc_word_sum[doc_idx] += 1 # 文档数量加上单词数量
            self.topic_word_sum[topic] += 1 # 对应主题单词数量加1

            self.doc_topic_count[doc_idx, topic] += 1 #文档-主题matrix
            self.word_topic_count[word_idx, topic] += 1 #主题-单词matrix
        print ('Initization Finished')

    def training(self,loop_time):
        start = time.time()
        print('第{}轮训练开始'.format(loop_time))
        for idx,(( word_idx,doc_idx, word_num), topic) in enumerate(zip(self.data, self.topic_att)):

            self.doc_topic_count[doc_idx, topic] -= 1  # 词在各topic的数量
            self.word_topic_count[word_idx, topic] -= 1  # 每个doc中topic的总数
            self.doc_word_sum[doc_idx] -= 1  # nwsum 每个topic词的总数
            self.topic_word_sum[topic] -= 1  # 每个doc中词的总数

            topic_doc = (self.doc_topic_count[doc_idx] + self.alpha) / (self.doc_word_sum[doc_idx] + self.k * self.alpha)  #
            word_topic = (self.word_topic_count[word_idx] + self.beta) / (self.topic_word_sum + self.words_num * self.beta)
            prob = topic_doc * word_topic
            prob = prob / sum(prob)

            new_topic = np.random.multinomial(1, prob, size=1)[0].argmax()

            self.doc_topic_count[doc_idx, new_topic] += 1  # 词在各topic的数量
            self.word_topic_count[word_idx, new_topic] += 1  # 每个doc中topic的总数
            self.doc_word_sum[doc_idx] += 1  # nwsum 每个topic词的总数
            self.topic_word_sum[new_topic] += 1  # 每个doc中词的总数

            self.topic_att[idx] = new_topic
        print ('Training Finished,duration{}'.format(time.time()-start))


    def cal_complexity(self):
        print('开始计算complexity')
        start = time.time()
        error_sum = 0
        doc_pb = (self.doc_topic_count + self.alpha) / (self.doc_word_sum + self.k * self.alpha).reshape(-1, 1)[0]
        word_pb = (self.word_topic_count + self.beta) / (self.topic_word_sum + self.words_num * self.beta)
        for  word_idx,doc_idx, word_num in self.data:
            error_sum += np.log(np.sum(doc_pb[doc_idx] * word_pb[word_idx]))
        loss = np.exp(-error_sum / len(self.data))
        duration = time.time() - start
        print ('Complexity finished {}'.format(duration))
        print (loss)
        self.loss_l.append(loss)

    def start(self,loop):
        print ('Starting')
        for i in range(loop):
            self.training(i)
            self.cal_complexity()





if __name__ == '__main__':
    tem = basic_lda()
    tem.read_dat('Data/TMN_processed.npy')
    tem.initize()
    tem.start(10)
    print (len(tem.data))