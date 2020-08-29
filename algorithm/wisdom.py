from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
import outlierDetection
import numpy as np
import util
from matplotlib import pyplot

training_data_path = '../input/training_data.csv'
training_labels_path = '../input/training_labels.csv'
test_data_path = '../input/test_data.csv'
test_label_path = '../input/test_labels.csv'
# training_data = np.array(util.build_dataframe(training_data_path))
# training_label = np.array(util.build_dataframe(training_labels_path))
test_data = np.array(util.build_dataframe(training_data_path))
test_label = np.array(util.build_dataframe(training_labels_path))

place1 = []
place3 = []
place4 = []
place5 = []
place6 = []
place7 = []
stan1 = []
stan3 = []
stan4 = []
stan5 = []
stan6 = []
stan7 = []
for i in range(len(test_label)):
    if test_label[i, 0] == 1:
        place1.append(i)
    elif test_label[i, 0] == 3:
        place3.append(i)
    elif test_label[i, 0] == 4:
        place4.append(i)
    elif test_label[i, 0] == 5:
        place5.append(i)
    elif test_label[i, 0] == 6:
        place6.append(i)
    elif test_label[i, 0] == 7:
        place7.append(i)

test_data1 = test_data[place1, :]
test_label1 = test_label[place1, :]
test_data3 = test_data[place3, :]
test_label3 = test_label[place3, :]
test_data4 = test_data[place4, :]
test_label4 = test_label[place4, :]
test_data5 = test_data[place5, :]
test_label5 = test_label[place5, :]
test_data6 = test_data[place6, :]
test_label6 = test_label[place6, :]
test_data7 = test_data[place7, :]
test_label7 = test_label[place7, :]

for line in test_data1:
    print("--------stat1----------")
    stan = 10000*np.nanstd(line)
    stan1.append(stan)
    print(stan)
for line in test_data3:
    print("--------stat3----------")
    stan = 10000*np.nanstd(line)
    stan3.append(stan)
    print(stan)
for line in test_data4:
    print("--------stat4----------")
    stan = 10000*np.nanstd(line)
    stan4.append(stan)
    print(stan)
for line in test_data5:
    print("--------stat5----------")
    stan = 10000*np.nanstd(line)
    stan5.append(stan)
    print(stan)
for line in test_data6:
    print("--------stat6----------")
    stan = 10000*np.nanstd(line)
    stan6.append(stan)
    print(stan)
for line in test_data7:
    print("--------stat7----------")
    stan = 10000*np.nanstd(line)
    stan7.append(stan)
    print(stan)

x1 = np.zeros(len(stan1)) + 1
x3 = np.zeros(len(stan3)) + 3
x4 = np.zeros(len(stan4)) + 4
x5 = np.zeros(len(stan5)) + 5
x6 = np.zeros(len(stan6)) + 6
x7 = np.zeros(len(stan7)) + 7

pyplot.scatter(x1, stan1, c='r', label='Type1')
pyplot.scatter(x3, stan3, c='b', label='Type3')
pyplot.scatter(x4, stan4, c='y', label='Type4')
pyplot.scatter(x5, stan5, c='g', label='Type5')
pyplot.scatter(x6, stan6, c='#672304', label='Type6')
pyplot.scatter(x7, stan7, c='c', label='Type7')
pyplot.xlabel('Type')
pyplot.ylabel('Std Distribution')
pyplot.legend()
pyplot.show()

"""画1类和4类的异常点数量
place4 = []
place1 = []
for i in range(len(test_label)):
    if test_label[i, 0] == 4:
        place4.append(i)
test_data4 = test_data[place4, :]
test_label4 = test_label[place4, :]
for i in range(len(test_label)):
    if test_label[i, 0] == 1:
        place1.append(i)
test_data1 = test_data[place1, :]
test_label1 = test_label[place1, :]
print(test_label1)
print(test_label4)

predict = []
count4 = np.zeros(150)
count1 = np.zeros(150)
x = np.arange(0, 150, 1)
for line in test_data4:
    err_count = outlierDetection.detect(line, 6)
    print("--------stat4----------")
    print(err_count)
    count4[err_count] += 1
    # stan = np.nanstd(line)
    # print(stan)

for line in test_data1:
    err_count = outlierDetection.detect(line, 6)
    print("--------stat1----------")
    print(err_count)
    count1[err_count] += 1

pyplot.plot(x, count1, 'r', label='Type1')
pyplot.plot(x, count4, 'b', label='Type4')
pyplot.xlabel('Number of Error Points')
pyplot.ylabel('Counting Number of that Error Points')
pyplot.legend()
pyplot.show()
"""

"""
    if 25 > err_count >= 3 and 1000 * abs(stan) < 0.4:
        predict.append(4)
    else:
        predict.append(0)

print('--------------------------')
print(predict)
print('--------------------------')
print(test_label)
print('--------------------------')
print(classification_report(y_true=test_label, y_pred=predict, zero_division=0))
"""
