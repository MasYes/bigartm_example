import artm
import uuid
from collections import Counter
from artm import BatchVectorizer, Dictionary, messages


def extract_unique_words(documents):
    words = set()
    for d in documents:
        for m in d:
            for w in d[m].split():
                words.add(w)
    return words


def convert_to_batch(documents, index=0, size=1000, dictionary=None):
    batch = artm.messages.Batch()
    batch.id = str(uuid.uuid4())

    # add all unique tokens (and their class_id) to batch.token and batch.class_id
    ut = {}
    for d in documents:
        for m in d:
            if m not in ut:
                ut[m] = set()
            for w in d[m].split():
                if dictionary is None or w in dictionary:
                    ut[m].add(w)

    ids = {}

    for class_id in sorted(ut):
        for token in sorted(ut[class_id]):
            ids[token + class_id] = len(ids)
            batch.token.append(token)
            batch.class_id.append(class_id)

    for i, d in enumerate(documents):
        item = batch.item.add()
        item.title = str(size*index + i)
        item.id = size*index + i
        for m in d:
            c = Counter(d[m].split())
            for w in c:
                if dictionary is None or w in dictionary:
                    item.token_id.append(ids[w + m])    # token_id refers to an index in batch.token
                    item.token_weight.append(c[w])

    return [batch]


def transform_batch(model, batch):
    batch_vectorizer = BatchVectorizer(batches=batch, process_in_memory_model=model)
    df = model.transform(batch_vectorizer=batch_vectorizer)
    return df.sort_index(axis=1).T.to_numpy()


def transform_batch_vectorizer(model, batch):
    df = model.transform(batch)
    return df.sort_index(axis=1).T.to_numpy()


def get_pwt(model, batch):
    batch_vectorizer = BatchVectorizer(batches=batch, process_in_memory_model=model)
    df = model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')
    df.columns = [batch[0].token[i] for i in batch[0].item[0].token_id]
    return df.T


def create_dictionary(batches):
    tf = Counter()
    df = Counter()
    for batch in batches:
        for item in batch.item:
            for freq, tid in zip(item.token_weight, item.token_id):
                cls, word = batch.class_id[tid], batch.token[tid]
                key = cls + "::" + word
                tf[key] += freq
                df[key] += 1
    global_n = sum(tf.values())
    dictionary = Dictionary()
    dictionary_data = messages.DictionaryData()
    dictionary_data.name = uuid.uuid1().urn.replace(':', '')
    # dictionary_data.
    for key in df.keys():
        cls, word = key.split('::')
        dictionary_data.token.append(word)
        dictionary_data.class_id.append(cls)
        dictionary_data.token_tf.append(tf[key])
        dictionary_data.token_df.append(df[key])
        dictionary_data.token_value.append(1.)

    dictionary.create(dictionary_data)
    return dictionary


def config_artm_logs():
    lc = artm.messages.ConfigureLoggingArgs()
    lc.log_dir = './logs'
    lib = artm.wrapper.LibArtm(logging_config=lc)
    lc.minloglevel = 0  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
    lib.ArtmConfigureLogging(lc)

