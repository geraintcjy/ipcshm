from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from outlierDetection import detect
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

exclude = [1, 2, 3, 4]
label_map = {5: 0, 6: 1, 7: 2}

del_lines = []
for index, line in enumerate(training_label):
    if line[0] in exclude:
        del_lines.append(index)
    else:
        line[0] = label_map[line[0]]
training_label = np.delete(training_label, del_lines, axis=0)
training_data = np.delete(training_data, del_lines, axis=0)

del_lines = []
for index, line in enumerate(test_label):
    if line[0] in exclude:
        del_lines.append(index)
    else:
        line[0] = label_map[line[0]]
test_label = np.delete(test_label, del_lines, axis=0)
test_data = np.delete(test_data, del_lines, axis=0)

training_label = to_categorical(training_label)

print(training_label)
training_label = to_categorical(training_label)
print(test_label)
print(test_data)

model = Sequential()
model.add(Dense(units=training_data.shape[1], activation='relu', input_dim=training_data.shape[1]))
model.add(Dense(units=len(label_map), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
training_data_scaler = scale(training_data, axis=1)
test_data_scaler = scale(test_data, axis=1)
val_label = to_categorical(test_label)
model.fit(training_data_scaler, training_label, epochs=50, batch_size=64, validation_data=(test_data_scaler, val_label))
classes = model.predict(test_data_scaler, batch_size=8)
predict = np.argmax(classes, axis=1)
predict_backup = predict

predict = predict_backup
print(predict)

for i, line in enumerate(test_data):
    # 2类型判断
    line_sorted = sorted(line, reverse=True)
    if abs(line_sorted[int(len(line_sorted) * 0.1)]) < 1e-3 and abs(line_sorted[int(len(line_sorted) * 0.9)]) < 1e-3:
        predict[i] = 2
        pass
    # 4类型判断
    else:
        pass
        # if np.array(test_label)[i][0] in [2,5,6,7]:
        #     continue
        # print(np.array(test_label)[i][0])
        # # detect(line, 4)
        # predict[i] = 4

print(classification_report(y_true=test_label, y_pred=predict, zero_division=0))
