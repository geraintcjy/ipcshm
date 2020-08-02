import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
"""
num代表第几天，应为0-30
csv最大列数16384，只能把72000放在行上
getDayData规模为72000*912，每列为一个小时的传感器数据
getDayLabel规模为1*912，每个数为一个小时的标签
"""

length = 15  # 数据简化的步长


def simplifier(data, length):
    # 取每length个数中绝对值最大的
    col_num = data.shape[1]
    answer = np.zeros((col_num, 72000 // length))
    for i in range(col_num):
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
    date = '2012-01-' + str(num + 1).zfill(2)
    data = np.zeros((72000, 912))
    for hour in range(24):
        datatemp = hdf5storage.loadmat('../' + date + '/' + date + ' ' + str(hour).zfill(2) + '-VIB.mat')['data']
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

def getDataLimited(num,simplify=True):
    '''每种类型获取同样数量的样本'''
    data = np.zeros((72000, num*7))
    wholeLabel = np.zeros((1, num*7))
    label = getLabel()
    count = np.zeros(7)
    col_index = 0
    finished = False
    for day in range(30):
        date = '2012-01-' + str(day + 1).zfill(2)
        if finished: break
        for hour in range(24):
            if finished: break
            datatemp = hdf5storage.loadmat('../' + date + '/' + date + ' ' + str(hour).zfill(2) + '-VIB.mat')['data']
            for j in range(38):
                cur_label=int(label[24 * day + hour, j])
                if count[cur_label-1]>=num:
                    continue
                data[:, col_index] = datatemp[:, j]
                wholeLabel[0][col_index] = cur_label
                count[cur_label-1] = count[cur_label-1]+1
                col_index = col_index+1
                if min(count)>=num: 
                    print('max day:{}'.format(day))
                    finished = True
                    break
    if simplify:
        data = simplifier(data, length)

    return data,wholeLabel


if __name__ == '__main__':
    # data = getDayData(1, True)
    # label = getDayLabel(1)
    data,label = getDataLimited(100)
    print(data)
    print(label)
    print(data.shape)
    print(label.shape)
    count=np.zeros(7)
    for x in label:
        for y in x:
            count[int(y)-1]=count[int(y)-1]+1
    print(count)
    # plot
    for test_id in np.random.choice(range(140),5):
        x = np.linspace(1, 3600, 72000 // length)
        y = []
        for rows in data:
            y.append(rows[test_id])
        plt.plot(x, y)
        print('当前类型{}'.format(label[0][test_id]))
        plt.show()
