import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from collections import Counter
import keras


class basic_lda():
    def __init__(self,alpha = 5,beta = 5,topic_num= 5,path = None):
        self.beta = beta
        self.alpha = alpha
        self.path = path
        self.dat = self.read_dat(path)
        self.k = topic_num

        #self.initlize()

    def read_dat(self):
        if self.path.endswith('.npz'):
            self.data = np.load(self.path)
            self.data_train = self.data['train']
            self.data_test = self.data['test']
            self.data_train[:, 0] = self.data_train[:, 0] - 1
            self.data_test[:, 0] = self.data_test[:, 0] - 1
            self.res = []
            for vec in self.data_train:
                if vec[-1] == 1:
                    self.res.append(vec)
                else:
                    new_vec = vec.copy()
                    new_vec[-1] = 1
                    for _ in range(vec[-1]):
                        self.res.append(new_vec)

        else:
            self.data = np.load(self.path)
            self.res = []
            for vec in self.data:
                if vec[-1] == 1:
                    self.res.append(vec)
                else:
                    new_vec = vec.copy()
                    new_vec[-1] = 1
                    for _ in range(vec[-1]):
                        self.res.append(new_vec)
