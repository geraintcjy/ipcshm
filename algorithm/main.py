import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import build_dataframe, make_training_data, feature_selection
from util import read_info_gain, compute_info_gains
from trainer import Trainer, normalization
from sklearn.metrics import classification_report


def read_dev_data():
    """
    Reads the subset of original training_data.csv.
    Used for development only
    """
    source = '../input-svm/training_data.csv'
    return pd.read_csv(source, index_col=0), None


def read_full_data(evaluate_features=False):
    """
    Read from the original assignment data and perform data pre-processing
    """
    training_data = '../input-svm/training_data.csv'
    training_labels = '../input-svm/training_labels.csv'
    test_data = '../input-svm/test_data.csv'

    training_data = make_training_data(training_data, training_labels)
    test_data = build_dataframe(test_data)

    if evaluate_features:
        print('***** recomputing info gain *****')
        feature_ig = compute_info_gains(training_data, save=True)
    else:
        print('***** reading cached info gain *****')
        feature_ig = read_info_gain()
    print('===== selecting useful features =====')
    feature_selection(training_data, feature_ig)
    feature_selection(test_data, feature_ig)

    return training_data, test_data


def save_predictions(predictions, test_data):
    df = pd.DataFrame(predictions, index=test_data.index)
    df.to_csv('../output/predicted_labels.csv', header=None)


def getAccuracy(predictions,label_path,test_data,showPlt=False):
    label=build_dataframe(label_path).values
    count, total = 0,0
    # pts=normalization(test_data.values)
    pts=test_data.values
    for index, prediction in enumerate(predictions):
        cur_label = label[index][0]
        if int(prediction) == int(cur_label):
            count = count + 1
            # print('序号{} 预测{} 实际{}'.format(index,prediction,cur_label))
            # x = np.linspace(1, 3600, 72000 // 15)
            # y = []
            # for pt in pts[index]:
            #     y.append(pt)
            # plt.plot(x, y)
            # plt.show()
        else:
            if showPlt and cur_label == 4:
                print('序号{} 预测{} 实际{}'.format(index, prediction, cur_label))

                x = np.linspace(1, 3600, 72000 // 15)
                y = []
                for pt in pts[index]:
                    y.append(pt)
                plt.plot(x, y)
                plt.show()
        total = total + 1
    print(count / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM Classifier')
    parser.add_argument('kernel', nargs='?', type=str, default='poly', choices=['linear', 'poly'],
                        help='The kernel function to use')
    parser.add_argument('strategy', nargs='?', type=str, default='one_vs_one', choices=['one_vs_one', 'one_vs_rest'],
                        help='The strategy to implement a multiclass SVM. Choose "one_vs_one" or "one_vs_rest"')
    parser.add_argument('C', nargs='?', type=float, default=1,
                        help='The regularization parameter that trades off margin size and training error')
    parser.add_argument('min_lagmult', nargs='?', type=float, default=0,
                        help='The support vector\'s minimum Lagrange multipliers value')
    parser.add_argument('cross_validate', nargs='?', type=bool, default=False,
                        help='Whether or not to cross validate SVM')
    parser.add_argument('evaluate_features', nargs='?', type=bool, default=False,
                        help='Will read the cache of feature evaluation results if set to False')
    parser.add_argument('mode', nargs='?', type=str, default='prod', choices=['dev', 'prod'],
                        help='Reads dev data in ../input-dev/ if set to dev mode, otherwise looks for datasets in ../input-svm/')
    config = vars(parser.parse_args())
    svm_params = {k: config[k] for k in ('kernel', 'strategy', 'C', 'min_lagmult')}

    if config['mode'] == 'dev':
        training_data, test_data = read_dev_data()
    elif config['mode'] == 'prod':
        training_data, test_data = read_full_data(config['evaluate_features'])
    print(training_data.shape, test_data.shape)
    trainer = Trainer(training_data, svm_params)

    if config['cross_validate']:
        trainer.cross_validate()
    else:
        print('===== training SVM units =====')
        trainer.train()
        print('===== predicting test data =====')
        # 预测测试集
        predictions = trainer.predict(test_data.values)
        for i, line in enumerate(test_data.values):
            line_sorted = sorted(line, reverse=True)
            if abs(line_sorted[int(len(line_sorted) * 0.1)]) < 1e-3 and abs(
                    line_sorted[int(len(line_sorted) * 0.9)]) < 1e-3:
                predictions[i] = 2

        save_predictions(predictions, test_data)
        getAccuracy(predictions, '../input-svm/test_labels.csv', test_data, True)
        print('------------------Report---------------------')
        original = labels = pd.read_csv('../input-svm/test_labels.csv', header=None).T
        print(classification_report(y_true=original, y_pred=predictions, zero_division=0))
        
        # 预测训练集
        # origin_data = build_dataframe('../input-svm/training_data.csv')
        # p2 = trainer.predict(origin_data.values)
        # for i,line in enumerate(origin_data.values):
        #     line_sorted=sorted(line,reverse=True)
        #     if abs(line_sorted[int(len(line_sorted)*0.01)]-np.mean(line))<1e-7 and abs(line_sorted[int(len(line_sorted)*0.99)]-np.mean(line))<1e-7:
        #         p2[i]=2
        # getAccuracy(p2,'../input-svm/training_labels.csv',origin_data,False)
