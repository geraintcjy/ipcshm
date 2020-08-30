from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from outlierDetection import detect
from outlierDetection import classification_6_7
import numpy as np
import pandas as pd
import os
import util

training_data_path = '../input/training_data.csv'
training_labels_path = '../input/training_labels.csv'
test_data_path = '../input/test_data.csv'
test_label_path = '../input/test_labels.csv'

training_data = np.array(util.build_dataframe(training_data_path))
training_label = np.array(util.build_dataframe(training_labels_path))
test_data = np.array(util.build_dataframe(test_data_path))
test_label = np.array(util.build_dataframe(test_label_path))

exclude = [2, 3, 4, 6]  # Exclude these types from Neural Network training
label_map = {}
whole = [1, 2, 3, 4, 5, 6, 7]
k = 0
for item in whole:
    if item not in exclude:
        label_map[item] = k
        k += 1
# Prejudging
print("---------------PreJudging---------------")
predict = np.zeros(test_label.shape[0], dtype=np.int)
temp = 0
for i, line in enumerate(test_data):
    print("Now Doing: ", temp + 1, "/912", sep='')
    temp += 1
    # Type 2 judging
    line_sorted = sorted(line, reverse=True)
    line_mean = np.nanmean(line_sorted)
    if abs(line_sorted[int(len(line_sorted) * 0.05)] - line_mean) / abs(line_mean) < 1e-3 and abs(
            line_sorted[int(len(line_sorted) * 0.95)] - line_mean) / abs(line_mean) < 1e-3:
        predict[i] = 2
    # Type 3 judging
    elif np.nanmax(line) - np.nanmin(line) < 4e-4:
        predict[i] = 3
    # Type 6 judging
    elif classification_6_7(line) == 6:
        predict[i] = 6
    # Type 4 judging
    else:
        err_count = detect(line, 8)
        stan = np.nanstd(line)
        if 25 >= err_count > 1 and 10000 * abs(stan) < 10.5:
            predict[i] = 4

print("---------------PreJudging Done---------------")

# Delete these excluding types
del_lines = []
for index, line in enumerate(training_label):
    if line[0] in exclude:
        del_lines.append(index)
    else:
        line[0] = label_map[line[0]]
training_label = np.delete(training_label, del_lines, axis=0)
training_data = np.delete(training_data, del_lines, axis=0)
"""
del_lines = []
for index, line in enumerate(test_label):
    if line[0] in exclude:
        del_lines.append(index)
    else:
        line[0] = label_map[line[0]]
test_label = np.delete(test_label, del_lines, axis=0)
test_data = np.delete(test_data, del_lines, axis=0)
"""
print(classification_report(y_true=test_label[:, 0], y_pred=predict, zero_division=0))
training_label = to_categorical(training_label)

# Neural Network
model = Sequential()
model.add(Dense(units=training_data.shape[1], activation='relu', input_dim=training_data.shape[1]))
model.add(Dense(units=len(label_map), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
training_data_scaler = scale(training_data, axis=1)
test_data_scaler = scale(test_data, axis=1)
val_label = to_categorical(test_label)
model.fit(training_data_scaler, training_label, epochs=50, batch_size=64, validation_data=(test_data_scaler, val_label))
classes = model.predict(test_data_scaler, batch_size=8)

inverse_label_map = dict(zip(label_map.values(), label_map.keys()))
for i in range(len(classes)):
    if predict[i] == 0:
        predict[i] = inverse_label_map[np.argmax(classes[i])]

print(predict)
print(classification_report(y_true=test_label, y_pred=predict, zero_division=0))
