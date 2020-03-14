import artm
import warnings

from bigartm_tools import convert_to_batch, transform_batch, create_dictionary, config_artm_logs
from transform_predictor import Predictor
from topics_for_classes import TFC

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
'''
Основная идея используемой хитрости - "зафиксировать" матрицу theta, она строилась не моделью, а
задавалась нами. Матрицу theta будем формировать таким образом, чтобы она соответствовала p(c|d).
В итоге, количество оптимальных тем намного ниже, чем в обычной Plsa, а точность выше.
'''
for data in datasets:
    print(f"|{data.name:^15}|", end='')
    data = data(lemmatize=True, exclude_stop_words=True, exclude_label_field=False)
    t = time.time()

    train_batch = convert_to_batch(data.train_docs)
    '''
    В этот раз при генерации тестовых батчей нужно указать какой-то большой индекс стартового документа. 
    Делается это потому, что в регуляризаторе мы будем указывать распределение p(c|d) документов по индексам. 
    Поэтому важно, чтобы тестовые документы имели индексы, которые не встречались в обучении, иначе регул.
    '''
    test_batch = convert_to_batch(data.test_docs, 1000000)

    dictionary = create_dictionary(train_batch)

    '''
    Класс, который я использую для того, чтобы формировать эту самую theta.
    Конструктор содержит два параметра. Первый - sizes, содержит количество тем, которое я выделяю каждому классу.
    Можно назначать разное число тем, но 4 - что-то более-менее адекватное.
    Второй параметр - число фоновых тем. Тут стоит 3, хотя обычно оптимально иметь 1 или даже 0.
    '''
    predictor = TFC([4] * len(set(data.train_labels)), 3)

    '''
    Генерируем эту самую матрицу theta.
    Метки класса трансформируются в векторы матрицы theta, что впоследствии будет использовано BigARTM 
    '''
    theta = []
    for i in data.train_labels:
        theta.append(predictor.convert_to_theta(int(i)))

    '''
    В этот раз число тем уже задано нами выше при инициализации TFC
    '''
    topic_num = predictor.topics - predictor.n_background  # the number of subject topics
    background_topic_num = predictor.n_background  # the number of background topic

    document_passes_num = 10
    processors_num = 12

    topics_names = ["subject_" + str(i) for i in range(topic_num)] + \
                   ["background_" + str(i) for i in range(background_topic_num)]  # назначаем имена темам

    subj_topics = topics_names[:topic_num]
    bgr_topics = topics_names[topic_num:]

    model = artm.ARTM(num_document_passes=document_passes_num,
                      num_topics=topic_num + background_topic_num,
                      topic_names=topics_names, num_processors=12,
                      seed=100)

    '''
    Тот самый регуляризатор, который будет связывать нашу theta с получаемой моделью.
    Через doc_titles указываем документы, которым соответствуют строки theta.
    '''
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='Theta', tau=10 ** 3, doc_topic_coef=theta,
                                                             doc_titles=[str(i)
                                                                         for i in range(len(data.train_docs))]))

    model.class_ids = {
        "title": 1,
        "text": 1,
        "label": 5,
    }
    model.initialize(dictionary.filter(min_df=10))

    model.fit_offline(batch_vectorizer=BatchVectorizer(batches=train_batch, process_in_memory_model=model),
                      num_collection_passes=3) # в этот раз, поскольку theta задана, 3 прохода вполне достаточно

    train_x = transform_batch(model, train_batch)
    test_x = transform_batch(model, test_batch)

    svm = LinearSVC()
    svm.fit(train_x, data.train_labels)

    predictor = Predictor(topic_num)
    predictor.fit(train_x, data.train_labels)

    t = time.time() - t

    '''
    Для оценки качества на обучающих документах приходится переделывать батч, чтобы модель не "подглядывала" в 
    theta по индексу документа.
    '''
    train_x = transform_batch(model, convert_to_batch(data.train_docs, 1000000))

    f1_train_svm = precision_recall_fscore_support(data.train_labels, svm.predict(train_x), average='macro')[2]
    f1_test_svm = precision_recall_fscore_support(data.test_labels, svm.predict(test_x), average='macro')[2]

    f1_train_tm = precision_recall_fscore_support(data.train_labels, predictor.predict(train_x), average='macro')[2]
    f1_test_tm = precision_recall_fscore_support(data.test_labels, predictor.predict(test_x), average='macro')[2]

    print(f"|{t:^15.2f}||{f1_train_svm:^15.2%}||{f1_test_svm:^15.2%}||{f1_train_tm:^15.2%}||{f1_test_tm:^15.2%}|")
    print(filler)