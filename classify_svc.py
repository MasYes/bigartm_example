import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import time

from datasets import AGNews, R8, R52, Ohsumed, NG20, DBPedia

warnings.filterwarnings("ignore")

datasets = [R8, R52, Ohsumed, AGNews, NG20, DBPedia]
title = "|{:^15}|" * (len(datasets) + 1)
filler = '-' * ((len(datasets) + 1) * 17)
print(filler)
print(title.format('', *[x.name for x in datasets]))
print(filler)
lemmatize = True
stopwords = True
print(f"|lemm:{str(lemmatize)[0:1]}, stop:{str(stopwords)[0:1]} |", end='')
for data in datasets:
    data = data(lemmatize=lemmatize, exclude_stop_words=stopwords, exclude_label_field=True)
    train_texts = []
    test_texts = []
    for doc in data.train_docs:
        train_texts.append(' '.join(doc.values()))
    for doc in data.test_docs:
        test_texts.append(' '.join(doc.values()))
    cv = CountVectorizer()
    tc_x = cv.fit_transform(train_texts)
    tf_idf = TfidfTransformer(use_idf=True)
    train_x = tf_idf.fit_transform(tc_x)
    test_x = tf_idf.transform(cv.transform(test_texts))
    if data.name == 'Eurlex':
        predictor = OneVsRestClassifier(LinearSVC(), n_jobs=12)
        mlb = MultiLabelBinarizer()
        train_y = mlb.fit_transform(data.train_labels)
        test_y = mlb.transform(data.test_labels)
    else:
        predictor = LinearSVC()
        train_y = data.train_labels
        test_y = data.test_labels
    t = time.time()
    predictor.fit(train_x, train_y)
    t = time.time() - t
    predictions = predictor.predict(test_x)
    f1 = precision_recall_fscore_support(test_y, predictions, average='macro')[2]
    print(f"|{f1:>8.2%}|{t:<6.1f}|", end='')
print(f'\n{filler}')

