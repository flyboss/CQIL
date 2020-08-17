import numpy as np


def SuccessRate_Avg(real, predict, num_results):
    predict = predict[:num_results]
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(real))


def Precision(real, predict, num_results):
    predict = predict[:num_results]
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(predict))


def MAP(real, predict, num_results):
    predict = predict[:num_results]
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))


def MRR(real, predict, num_results):
    predict = predict[:num_results]
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def get_metrics_scores(real, predict):
    metrics_scores = []
    metrics_scores.append(SuccessRate_Avg(real, predict, 1))
    metrics_scores.append(SuccessRate_Avg(real, predict, 5))
    metrics_scores.append(SuccessRate_Avg(real, predict, 10))

    metrics_scores.append(Precision(real, predict, 1))
    metrics_scores.append(Precision(real, predict, 5))
    metrics_scores.append(Precision(real, predict, 10))

    metrics_scores.append(MAP(real, predict, 1))
    metrics_scores.append(MAP(real, predict, 5))
    metrics_scores.append(MAP(real, predict, 10))

    metrics_scores.append(MRR(real, predict, 10))
    return metrics_scores


def metrics_scores_to_dict(metrics_scores):
    scores_name = ['SuccessRate_1', 'SuccessRate_5', 'SuccessRate_10',
                   'Precision_1', 'Precision_5', 'Precision_10',
                   'MAP_1', 'MAP_5', 'MAP_10', 'MRR']
    scores_list = [[] for _ in scores_name]

    for scores in metrics_scores:
        for index in range(len(scores_name)):
            scores_list[index].append(scores[index])

    metrics_dict = {}
    for index, name in enumerate(scores_name):
        metrics_dict[name] = np.mean(scores_list[index])

    return metrics_dict
