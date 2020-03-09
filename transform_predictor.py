import numpy as np
from collections import Counter

class Predictor:

    def __init__(self, topic_num = None):
        self.weights = {}
        self.size = Counter()
        self.topic_num = topic_num

    def fit(self, X, y):
        if self.topic_num == None:
            self.topic_num = len(X[0])
        for feat, label in zip(X, y):
            feat = np.array(feat[:self.topic_num])
            if label not in self.weights:
                self.weights[label] = np.zeros(feat.shape)
            self.weights[label] += feat
            self.size[label] += 1
        for k in self.weights:
            self.weights[k] = self.weights[k] / np.sum(self.weights[k])

    def predict(self, X, y=None):
        predictions = []
        for feat in X:
            predictions.append(sorted(self.weights.keys(), key=lambda x: np.dot(self.weights[x], feat[:self.topic_num]),
                                      reverse=True)[0])
        return predictions
