import getData
import matplotlib.pyplot as plt
import numpy as np

# dayData规模为912*72000，每行为一个小时的传感器数据
# dayLabel规模为912*1，每个数为一个小时的标签
day0Data, day0Label = getData.getDay(0)

y = day0Data[0]
x = np.linspace(1, 3600, 72000)
plt.plot(x, y)
plt.show()

# model
'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(72000, input_dim=72000, activation='relu'))
model.add(Dense(72000, activation='relu'))
model.add(Dense(24000, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
'''

print(day0Data.shape)
print(day0Label.shape)
print(day0Data)
print(day0Label)
