import artm
import warnings

from bigartm_tools import convert_to_batch, transform_batch, create_dictionary, config_artm_logs
from transform_predictor import Predictor
from artm import BatchVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import time

from datasets import AGNews, R8, R52, Ohsumed, NG20, DBPedia

warnings.filterwarnings("ignore")
config_artm_logs()

title = "|{:^15}|" * 6
filler = '-' * 6 * 17
print(filler)
print(title.format('', "Training time", "Train SVM", "Test SVM", "Train TM", "Test TM"))
print(filler)
datasets = [R8, R52, Ohsumed, AGNews, NG20, DBPedia]

for data in datasets:
    print(f"|{data.name:^15}|", end='')
    data = data(lemmatize=True, exclude_stop_words=True, exclude_label_field=False)
    t = time.time()
    train_batch = convert_to_batch(data.train_docs)
    test_batch = convert_to_batch(data.test_docs)
    dictionary = create_dictionary(train_batch)

    topic_num = 100  # количество смысловых тем, в которых находятся важные слова
    background_topic_num = 3  # количество "фоновых" тем, в которые мы будем помещать бессмысленные слова (стоп-слова)
    document_passes_num = 10  # количество проходов по документу внутри одного E-шага
    processors_num = 12

    topics_names = ["subject_" + str(i) for i in range(topic_num)] + \
                   ["background_" + str(i) for i in range(background_topic_num)]  # назначаем имена темам

    subj_topics = topics_names[:topic_num]
    bgr_topics = topics_names[topic_num:]

    model = artm.ARTM(num_document_passes=document_passes_num,
                      num_topics=topic_num + background_topic_num,
                      topic_names=topics_names,
                      seed=100,  # helps to get stable results
                      num_processors=processors_num)

    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='Decorrelator', tau=10 ** 4))  # обычный декоррелятор
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothTheta',
                                                            topic_names=bgr_topics,
                                                            tau=0.3))  # сглаживаем Theta для фоновых тем
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                            topic_names=subj_topics,
                                                            tau=-0.3))  # разреживаем Theta для "хороших" тем
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhi',
                                                            topic_names=bgr_topics,
                                                            class_ids=["text"],
                                                            tau=0.1))  # сглаживаем Theta для фоновых тем
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi',
                                                            topic_names=subj_topics,
                                                            class_ids=["text"],
                                                            tau=-0.1))    # разреживаем Theta для "хороших" тем
#    model.regularizers.add(artm.LabelRegularizationPhiRegularizer(class_ids=["label"]))    # этот регуляризатор мало у кого дает
#    # хороший результат, но ты попробуй :) у меня он вылетает с ошибкой :(

    model.class_ids = {
        "title": 1,
        "text": 1,
        "label": 5,
    }
    model.initialize(dictionary.filter(min_df=10))
    model.fit_offline(batch_vectorizer=BatchVectorizer(batches=train_batch, process_in_memory_model=model),
                      num_collection_passes=3)
    train_x = transform_batch(model, train_batch)
    test_x = transform_batch(model, test_batch)

    svm = LinearSVC()
    svm.fit(train_x, data.train_labels)

    predictor = Predictor(topic_num)
    predictor.fit(train_x, data.train_labels)

    t = time.time() - t
    f1_train_svm = precision_recall_fscore_support(data.train_labels, svm.predict(train_x), average='macro')[2]
    f1_test_svm = precision_recall_fscore_support(data.test_labels, svm.predict(test_x), average='macro')[2]

    f1_train_tm = precision_recall_fscore_support(data.train_labels, predictor.predict(train_x), average='macro')[2]
    f1_test_tm = precision_recall_fscore_support(data.test_labels, predictor.predict(test_x), average='macro')[2]

    print(f"|{t:^15.2f}||{f1_train_svm:^15.2%}||{f1_test_svm:^15.2%}||{f1_train_tm:^15.2%}||{f1_test_tm:^15.2%}|")
    print(filler)