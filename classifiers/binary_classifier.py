'''
Simple wrapper of a binary classifier
'''

import scipy
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.preprocessing import StandardScaler
from enum import Enum
from scipy.sparse import csr_matrix

from utils.arithmetic_utils import pointwise_mult, mat_concat

class CLASSIFIER_TYPE(Enum):
    PA = 1
    LR = 2
    SVM = 3


class BinaryClassifier(object):

    def __init__(self, classifier_type, scale_features=True):
        self.scale_features = scale_features
        self.classifier_type = classifier_type
        self.clear()
        self.train_std = 0
        self.random_gen = np.random.RandomState(136543785)

    def clear(self, remember_train_std_if_supported=False):
        self.positive_instances = []
        self.negative_instances = []
        # self.classifier = svm.SVC(kernel='linear')
        if self.classifier_type == CLASSIFIER_TYPE.LR :
            #print (CLASSIFIER_TYPE.LR)
            self.classifier = LogisticRegression(C=1.0)
        elif self.classifier_type == CLASSIFIER_TYPE.PA :
            #print (CLASSIFIER_TYPE.PA)
            self.classifier = PassiveAggressiveClassifier(loss='hinge',C=1.0)
        elif self.classifier_type == CLASSIFIER_TYPE.SVM :
            self.classifier = svm.SVC(kernel='linear')
        if not remember_train_std_if_supported:
            self.train_std = 0

    def add_positive_instances(self, positive_instances):
        self.positive_instances.extend(positive_instances)

    def add_negative_instances(self, negative_instances):
        self.negative_instances.extend(negative_instances)

    def train(self):
        X = self.positive_instances + self.negative_instances
        y = np.asarray([1] * len(self.positive_instances) + [0] * len(self.negative_instances))

        # shuffling the train instances in case classifier is sensitive to this order
        Xy = list(zip(X,y))
        self.random_gen.shuffle(Xy)
        X[:], y[:] = zip(*Xy)

        X = mat_concat(X)

        if self.scale_features:
            if self.train_std == 0:
                self.train_std = (pointwise_mult(X,X).mean() - X.mean()**2)**0.5
            X = X / self.train_std
        # X = X/self.train_std
        self.classifier.fit(X,y)

    def predict(self, instances):
        # scaled_instances = [inst/self.train_std for inst in instances]
        instances = mat_concat(instances)
        if self.scale_features and self.train_std > 0:
            instances = instances/self.train_std
        return self.classifier.predict(instances)

if __name__ == '__main__':

    import numpy as np
    X = [np.ndarray([0, 0]), np.ndarray([1, 1])]
    y = [0, 1]

    pos_vecs = [np.asarray([1,1]),np.asarray([2,2])]
    neg_vecs = [np.asarray([-1,-1]),np.asarray([-2,-2])]

    cls = BinaryClassifier()
    cls.add_positive_instances(pos_vecs)
    cls.add_negative_instances(neg_vecs)
    cls.train()
    print(cls.predict(neg_vecs))

