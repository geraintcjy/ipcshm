import multiprocessing
import os
import pickle
import queue
from collections import Counter

import numpy as np
import util
from confusion_matrix import ConfusionMatrix
from svm import SVM
from util import folds_indexes, accuracy_score
from util import timing, setup_tmp

num_cpus = multiprocessing.cpu_count()


def make_svm_unit(params):
    """
    Worker function that trains a SVM unit:
    SVM params are dumped to a temporary pickle file and the worker informs
    its master process where the pickle is.
    """
    _id, pair, X, Y, config, mailbox = params['_id'], params['pair'], params['X'], params['Y'], params['config'], \
                                       params['mailbox']
    unit = SVMUnit(pair, X, Y, config)
    fpath = os.path.join('./tmp', str(_id) + '.pkl')
    with open(fpath, 'wb') as f: pickle.dump(unit, f)
    mailbox.put(fpath)


def normalization(X):
    """
    把输入数据标准化
    """
    new_X=[]
    for line in X:
        mu = np.mean(line, axis=0)
        sigma = np.std(line, axis=0)
        if sigma>1e-7:
            line = (line - mu) / sigma
        new_X.append(line)
    return np.array(new_X)
    # return X


def fast_predict(X, pair, predictions):
    '''根据图形特征快速预测'''
    # if [3] in pair:
    #     data=predictions
    #     for i,line in enumerate(X):
    #         _range = np.max(line) - np.min(line)
    #         if _range<5e-4:
    #             data[i]=1 if 3==pair[0][0] else -1
    #     return data
    return predictions


class SVMUnit:
    """
    Represents a SVM unit in a multiclass SVM classifier.
    Using one-vs-one implementation strategy, each unit classifies a pair of classes.
    Using one-vs-rest implementation strategy, each unit classifies a class and all the other classes.
    """

    def __init__(self, pair, X, Y, config):
        self.class_lookup = dict({str(kls): 1. for kls in pair[0]}, **{str(kls): -1. for kls in pair[1]})
        self.prediction_lookup = {1.: pair[0], -1.: pair[1]}
        self.strategy = config['strategy']
        self.svm = SVM(**config).fit(normalization(X), self.transform_y(Y))
        self.pair = pair

    def lookup_predictions(self, preds):
        """
        :param preds: an array of 1. or -1. SVM predictions
        :return: corresponding classes of 1 or -1.
        """
        if self.strategy == 'one_vs_one':
            vfunc = np.vectorize(lambda val: self.prediction_lookup[val][0])
            return vfunc(preds)

        if self.strategy == 'one_vs_rest':
            vfunc = np.vectorize(lambda val: self.prediction_lookup[1.][0])
            return vfunc(preds)

    def transform_y(self, Y):
        """
        convert labels to (-1, 1)
        """
        vfunc = np.vectorize(lambda kls: self.class_lookup[str(kls)])
        return vfunc(Y)

    def getPair(self):
        return self.pair


class Trainer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.svm_units = []

    @setup_tmp
    @timing
    def train(self, data=None):
        """
        Train a multi-class SVM classifier by creating many One-vs-N SVM units.
        Parallelise the training for optimising the performance
        """

        def create_packets(pairs, mailbox):
            ret = []
            for idx, p in enumerate(pairs):
                subset = data.loc[p[0] + p[1]]
                ret.append({
                    '_id': idx,
                    'pair': p,
                    'X': subset.values,
                    'Y': subset.index.values,
                    'config': self.config,
                    'mailbox': mailbox
                })
            return ret

        if data is None:
            data = self.data

        # get all one-vs-N pairs
        classes = np.unique(data.index.values)
        pairs = getattr(util, self.config['strategy'] + '_pairs')(classes)

        # create worker processes
        num_workers = min(num_cpus, len(pairs))
        pool = multiprocessing.Pool(num_workers)
        pmanager = multiprocessing.Manager()
        mailbox = pmanager.Queue()

        # workers starting working on making svm units
        # 可能是数据中的nan导致的，先做数据简化，顺便处理nan数值
        pool.map(make_svm_unit, create_packets(pairs, mailbox))

        # each time a worker returns the pickle path to the trained SVM unit
        while True:
            try:
                fpath = mailbox.get(timeout=0.05)
                with open(fpath, 'rb') as f:
                    self.svm_units.append(pickle.load(f))
            except queue.Empty:
                break

        pool.close()
        pool.join()
        print('done')

    @timing
    def cross_validate(self, data=None, num_folds=10):
        """
        Split the training data into N folds. Use N-1 folds as training data
        and 1 fold as the testing data during each iteration.

        Outputs the confusion matrix and other accuracy measurement stats the end of N-CV.
        """
        if data is None:
            data = self.data
        if len(data) < num_folds:
            raise ValueError('Not enough data to make {} folds'.format(num_folds))

        folds = folds_indexes(data, num_folds)
        folds_accuracy = []
        cm = ConfusionMatrix(np.unique(data.index.values))

        for i, (train_idx, test_idx) in enumerate(folds):
            # find the indexes of other folds data in the dataframe
            print('Fold {}'.format(i))
            training_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            self.train(training_data)
            # predict and evaluate
            predictions = self.predict(test_data.values)
            real_vals = test_data.index.values
            accuracy = accuracy_score(predictions, real_vals)
            cm.update(predictions, real_vals)
            # clean up
            folds_accuracy.append(accuracy)
            self.svm_units = []
            print(accuracy)

        print('Mean Accuracy : {}'.format(np.mean(folds_accuracy)))
        print('***** Detailed Summary *****')
        print(cm.statistics())
        print('***** Confusion Matrix *****')
        print(cm)
        cm.save(folder='../experiments/')
        return self

    def __one_vs_one_predict__(self, X):
        """
        Predict a sample data's class by getting the major vote over all pairwise SMV units
        """
        unit_preds = []
        n = len(X)
        for unit in self.svm_units:
            fast_pd = fast_predict(X, unit.pair, unit.svm.predict(X))
            if str(fast_pd[0]) == 'nan':
                a = 0
            pred = unit.lookup_predictions(fast_pd)
            unit_preds.append(pred)
        unit_preds = np.array(unit_preds)
        return [Counter(unit_preds[:, i]).most_common(1)[0][0] for i in range(n)]

    def __one_vs_rest_predict__(self, X):
        """
        The class of a data point is whichever class has a decision
        function with highest value, regardless of sign
        """
        n = len(X)
        unit_margins = np.array([unit.svm.predict(X) for unit in self.svm_units])
        idx_max = [np.argmax(unit_margins[:, i]) for i in range(n)]
        predictions = []

        for i in range(n):
            best = idx_max[i]
            predictions.append(str(self.svm_units[best].lookup_predictions(unit_margins[best, i])))

        return predictions

    def predict(self, X):
        return {
            'one_vs_one': self.__one_vs_one_predict__,
            'one_vs_rest': self.__one_vs_rest_predict__
        }.get(self.config['strategy'])(normalization(X))
