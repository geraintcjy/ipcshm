from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from outlierDetection import detect
import numpy as np
import pandas as pd
import os
import util

training_data_path = '../input/training_data.csv'
training_labels_path = '../input/training_labels.csv'
test_data_path = '../input/test_data.csv'
test_label_path = '../input/test_labels.csv'

training_data = util.build_dataframe(training_data_path)
training_label = util.build_dataframe(training_labels_path)
test_data = util.build_dataframe(test_data_path)
test_label = util.build_dataframe(test_label_path)
training_label -= 1
training_label = to_categorical(training_label)

# model = Sequential()
# model.add(Dense(units=training_data.shape[1], activation='relu', input_dim=training_data.shape[1]))
# model.add(Dense(units=training_data.shape[1], activation='relu'))
# model.add(Dense(units=training_data.shape[1] // 2, activation='relu'))
# model.add(Dense(units=7, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

# model.fit(training_data, training_label, epochs=100, batch_size=5)
# classes = model.predict(test_data, batch_size=5)
# predict = np.argmax(classes, axis=1)
# predict_backup = predict

# print(predict_backup)
# predict = predict_backup
test_data = np.array(test_data)
for i,line in enumerate(test_data):
    # 2类型判断
    line_sorted = sorted(line, reverse=True)
    if abs(line_sorted[int(len(line_sorted) * 0.1)]) < 1e-3 and abs(line_sorted[int(len(line_sorted) * 0.9)]) < 1e-3:
        # predict[i] = 2
        pass
    # 4类型判断
    else:
        if np.array(test_label)[i][0] in [2,3,5,6,7]:
            continue
        print(np.array(test_label)[i][0])
        detect(line)
        # predict[i] = 4

print(classification_report(y_true=test_label, y_pred=predict, zero_division=0))
