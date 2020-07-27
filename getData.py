import hdf5storage
import numpy as np
import matplotlib.pyplot as plt


def getData(date):
    # data获取date一天24小时的数据
    data = []
    for hour in range(24):
        if hour < 10:
            datatemp = hdf5storage.loadmat(date + '/' + date + ' 0' + str(hour) + '-VIB.mat')['data']
        else:
            datatemp = hdf5storage.loadmat(date + '/' + date + ' ' + str(hour) + '-VIB.mat')['data']
        data.append(datatemp)

    return data


def getLabel():
    return hdf5storage.loadmat('label_r.mat')['manual_r']


def getDay(num):  # num代表第几天，应为0-30
    # wholeData规模为912*72000，每行为一个小时的传感器数据
    # wholeLabel规模为912*1，每个数为一个小时的标签
    wholeData = np.zeros((912, 72000))  # 完整的数据
    wholeLabel = np.zeros((912, 1))  # 完整的标签
    label = getLabel()
    if num < 9:
        da = getData('2012-01-0' + str(num + 1))
    else:
        da = getData('2012-01-' + str(num + 1))
    """
    次序：
    00:00-01:00的1,2,3,4,5...测点数据,
    01:00-02:00的1,2,3,4,5...测点数据,
    02:00-03:00的1,2,3,4,5...测点数据,
    ...
    """
    for j in range(24):
        for k in range(38):
            wholeData[j * 38 + k] = da[j][:, k]
            wholeLabel[j * 38 + k] = label[24 * num + j, k]
    return wholeData, wholeLabel


if __name__ == '__main__':
    data = getData('2012-01-02')
    label = getLabel()
    print(data)
    print(label)

    """
    # plot
    x = np.linspace(1, 3600, 72000)
    y = []
    for rows in data[0]:
        y.append(rows[0])
    plt.scatter(x, y)
    plt.show()
    """
