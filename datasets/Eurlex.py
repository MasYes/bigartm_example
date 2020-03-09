from .common import ReportedResults
from utils import clear_text
import json
import io

class Eurlex:
    name = "Eurlex"

    def __init__(self, ngrammer=None, lemmatize=True, exclude_stop_words=True, exclude_label_field=True):
        self.reported_results = ReportedResults()
        self.train_docs = []
        self.train_labels = []
        self.test_docs = []
        self.test_labels = []
        self.name = "Eurlex"
        path = f'./data/eurolex/'

        for line in io.open(path + ('train_lemm.txt' if lemmatize else 'train.txt'), encoding='utf-8', errors='ignore'):
            doc = json.loads(line)
            labels = " ".join(doc["concepts"])
            self.train_labels.append(doc["concepts"])
            doc["label"] = labels
            doc["text"] = clear_text(" ".join(doc["main_body"]).lower(), exclude_stop_words=exclude_stop_words)
            doc["header"] = clear_text(doc["header"].lower(), exclude_stop_words=exclude_stop_words)
            doc["recitals"] = clear_text(doc["recitals"].lower(), exclude_stop_words=exclude_stop_words)
            doc["attachments"] = clear_text(doc["attachments"].lower(), exclude_stop_words=exclude_stop_words)
            del doc["celex_id"]
            del doc["uri"]
            del doc["main_body"]
            del doc["concepts"]
            del doc["title"]
            if ngrammer is not None:
                text, ng = ngrammer.greedy_ngrams(doc["text"])
                doc["text"] = text
                doc["ngrams"] = ng
            if exclude_label_field:
                del doc["label"]
            self.train_docs.append(doc)

        for line in io.open(path + ('test_lemm.txt' if lemmatize else 'test.txt'), encoding='utf-8', errors='ignore'):
            doc = json.loads(line)
            labels = " ".join(doc["concepts"])
            self.test_labels.append(doc["concepts"])
            doc["label"] = labels
            doc["text"] = clear_text(" ".join(doc["main_body"]).lower(), exclude_stop_words=exclude_stop_words)
            doc["header"] = clear_text(doc["header"].lower(), exclude_stop_words=exclude_stop_words)
            doc["recitals"] = clear_text(doc["recitals"].lower(), exclude_stop_words=exclude_stop_words)
            doc["attachments"] = clear_text(doc["attachments"].lower(), exclude_stop_words=exclude_stop_words)
            del doc["celex_id"]
            del doc["uri"]
            del doc["main_body"]
            del doc["concepts"]
            del doc["title"]
            if ngrammer is not None:
                text, ng = ngrammer.greedy_ngrams(doc["text"])
                doc["text"] = text
                doc["ngrams"] = ng
            del doc["label"]
            self.test_docs.append(doc)

