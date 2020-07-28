import hdf5storage
import numpy as np
import matplotlib.pyplot as plt


def getDayData(num):
    # getDayData获取date一天24小时的数据
    """
    列次序：
    00:00-01:00的1,2,3,4,5...测点数据,
    01:00-02:00的1,2,3,4,5...测点数据,
    02:00-03:00的1,2,3,4,5...测点数据,
    ...
    """
    if num < 9:
        date = '2012-01-0' + str(num + 1)
    else:
        date = '2012-01-' + str(num + 1)
    data = np.zeros((72000, 912))
    for hour in range(24):
        if hour < 10:
            datatemp = hdf5storage.loadmat('../'+date + '/' + date + ' 0' + str(hour) + '-VIB.mat')['data']
        else:
            datatemp = hdf5storage.loadmat('../'+date + '/' + date + ' ' + str(hour) + '-VIB.mat')['data']
        for j in range(38):
            data[:, hour * 38 + j] = datatemp[:, j]

    return data


def getLabel():
    return hdf5storage.loadmat('../label_r.mat')['manual_r']


"""
num代表第几天，应为0-30
csv最大列数16384，只能把72000放在行上
wholeData规模为72000*912，每列为一个小时的传感器数据
wholeLabel规模为1*912，每个数为一个小时的标签
"""


def getDayLabel(num):
    wholeLabel = np.zeros((1, 912))  # 完整的标签
    label = getLabel()
    for j in range(24):
        for k in range(38):
            wholeLabel[0][j * 38 + k] = label[24 * num + j, k]
    return wholeLabel


if __name__ == '__main__':
    data = getDayData(1)
    label = getDayLabel(1)
    print(data)
    print(label)
    print(data.shape)
    print(label.shape)

    """
    # plot
    x = np.linspace(1, 3600, 72000)
    y = []
    for rows in data[0]:
        y.append(rows[0])
    plt.scatter(x, y)
    plt.show()
    """
