from .common import ReportedResults
from utils import clear_text
import unidecode
import io
import bz2
import json

class DBPedia:
    name = "DBPedia"

    def __init__(self, ngrammer=None, lemmatize=True, exclude_stop_words=True, exclude_label_field=True):
        self.reported_results = \
            ReportedResults().add_result('BERT UDA', 98.91, 'https://paperswithcode.com/sota/text-classification-on-ag-news')\
                .add_result('BERT', 99.32, 'https://paperswithcode.com/sota/text-classification-on-ag-news')\
                .add_result('XLNet', 99.38, 'https://paperswithcode.com/sota/text-classification-on-ag-news')\
                .add_result('fastText', 98.6, 'https://paperswithcode.com/sota/text-classification-on-ag-news')
        self.train_docs = []
        self.train_labels = []
        self.test_docs = []
        self.test_labels = []
        self.name = "DBPedia"
        if not lemmatize:
            path = f'./gdrive/{self.name.lower()}/original_text/'
        elif not exclude_stop_words:
            path = f'./gdrive/{self.name.lower()}/lemmatized/'
        else:
            path = f'./gdrive/{self.name.lower()}/lemmatized_wo_stopwords/'

        for line in bz2.open(path + 'train.bz2', 'rt', encoding='utf-8'):
            doc = json.loads(line)
            doc["label"] = str(doc["label"])
            if ngrammer is not None:
                text, ng = ngrammer.greedy_ngrams(doc["text"])
                doc["text"] = text
                doc["ngrams"] = ng
            self.train_labels.append(doc["label"])
            if exclude_label_field:
                del doc["label"]
            self.train_docs.append(doc)

        for line in bz2.open(path + 'test.bz2', 'rt', encoding='utf-8'):
            doc = json.loads(line)
            doc["label"] = str(doc["label"])
            if ngrammer is not None:
                text, ng = ngrammer.greedy_ngrams(doc["text"])
                doc["text"] = text
                doc["ngrams"] = ng
            self.test_labels.append(doc["label"])
            del doc["label"]
            self.test_docs.append(doc)