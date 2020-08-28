from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import util

training_data_path = '../input/training_data.csv'
training_labels_path = '../input/training_labels.csv'
test_data_path = '../input/test_data.csv'
test_label_path = '../input/test_labels.csv'

exclude = [2, 3, 5, 6, 7]
train_seq = []
test_seq = []
training_data = build_dataframe(training_data_path)
training_label = build_dataframe(training_labels_path)
test_data = build_dataframe(test_data_path)
test_label = build_dataframe(test_label_path)
for i in range(training_label.shape[1]):
    if int(training_label.iat[0, i]) not in exclude:
        train_seq.append(i)
for i in range(test_label.shape[1]):
    if int(test_label.iat[0, i]) not in exclude:
        test_seq.append(i)
training_data = training_data.iloc[:, train_seq]
training_label = training_label.iloc[:, train_seq]
test_data = test_data.iloc[:, test_seq]
test_label = test_label.iloc[:, test_seq]
training_label -= 1
print(training_data)
print(training_label)
training_label = to_categorical(training_label)
print(test_label)
print(test_data)

model = Sequential()
model.add(Dense(units=training_data.shape[1], activation='relu', input_dim=training_data.shape[1]))
model.add(Dense(units=training_data.shape[1], activation='relu'))
model.add(Dense(units=training_data.shape[1] // 2, activation='relu'))
model.add(Dense(units=7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

model.fit(training_data, training_label, epochs=100, batch_size=5)
classes = model.predict(test_data, batch_size=5)
predict = np.argmax(classes, axis=1)
predict_backup = predict

print(predict_backup)
predict = predict_backup
for i in range(len(test_data)):
    # 2类型判断
    if test_data.iat[i, 1] < 0.01:
        predict[i] = 2
    # 4类型判断
    elif np.max(test_data.iloc[i, 2:]) / test_data.iat[i, 1] > 11 and test_data.iat[i, 1] < 0.08 and np.max(
            test_data.iloc[i, 2:]) < 1:
        predict[i] = 4

print(classification_report(y_true=test_label, y_pred=predict, zero_division=0))
