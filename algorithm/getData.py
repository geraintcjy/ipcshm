import hdf5storage
import numpy as np
import matplotlib.pyplot as plt

"""
num代表第几天，应为0-30
csv最大列数16384，只能把72000放在行上
getDayData规模为72000*912，每列为一个小时的传感器数据
getDayLabel规模为1*912，每个数为一个小时的标签
"""

length = 15  # 数据简化的步长


def simplifier(data, length):
    # 取每length个数中绝对值最大的
    answer = np.zeros((912, 72000 // length))
    for i in range(912):
        maxnum = 0.  # 记录最大值
        count = 0  # 控制length
        new_column = np.zeros(72000 // length)

        while count <= 72000:
            if count % length == 0 and count > 0:
                if count == 72000:
                    pass
                else:
                    try:
                        if abs(maxnum) < abs(data[count][i]):
                            maxnum = data[count][i]
                    except RuntimeError:
                        if str(data[count][i]) == 'NaN':
                            data[count][i] = 0

                new_column[count // length - 1] = maxnum
                maxnum = 0.
                count += 1
            else:
                try:
                    if abs(maxnum) < abs(data[count][i]):
                        maxnum = data[count][i]
                except RuntimeError:
                    if str(data[count][i]) == 'NaN':
                        data[count][i] = 0
                count += 1

        answer[i] = new_column
    return answer.T


def getDayData(num, simplify):
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
            datatemp = hdf5storage.loadmat('../' + date + '/' + date + ' 0' + str(hour) + '-VIB.mat')['data']
        else:
            datatemp = hdf5storage.loadmat('../' + date + '/' + date + ' ' + str(hour) + '-VIB.mat')['data']
        for j in range(38):
            data[:, hour * 38 + j] = datatemp[:, j]

    if simplify:
        data = simplifier(data, length)

    return data


def getLabel():
    return hdf5storage.loadmat('../label_r.mat')['manual_r']


def getDayLabel(num):
    wholeLabel = np.zeros((1, 912))
    label = getLabel()
    for j in range(24):
        for k in range(38):
            wholeLabel[0][j * 38 + k] = label[24 * num + j, k]
    return wholeLabel


if __name__ == '__main__':
    data = getDayData(1, True)
    label = getDayLabel(1)
    print(data)
    print(label)
    print(data.shape)
    print(label.shape)

    # plot
    x = np.linspace(1, 3600, 72000 // length)
    y = []
    for rows in data:
        y.append(rows[0])
    plt.plot(x, y)
    plt.show()
