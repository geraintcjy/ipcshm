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


length = 8  # 数据简化的步长
exclude = []  # 忽略2类



def simplifier1(data, length):
    """取每length个数中绝对值最大的"""
    col_num = data.shape[1]
    answer = np.zeros((col_num, 72000 // length))
    for i in range(col_num):
        maxnum = 0.  # 记录最大值
        count = 0  # 控制length
        sum = 0  # 求和
        new_column = np.zeros(72000 // length)

        while count <= 72000:
            if count % length == 0 and count > 0:
                if count == 72000:
                    pass
                else:
                    cur = data[count][i]
                    sum += cur
                    try:
                        if abs(maxnum) < abs(cur):
                            maxnum = cur
                    except RuntimeError:
                        if str(cur) == 'NaN':
                            data[count][i] = sum / count

                new_column[count // length - 1] = maxnum
                maxnum = 0.
                count += 1
            else:
                cur = data[count][i]
                sum += cur
                try:
                    if abs(maxnum) < abs(cur):
                        maxnum = cur
                except RuntimeError:
                    if str(cur) == 'NaN':
                        data[count][i] = sum / count
                count += 1

        answer[i] = new_column

    return answer.T


def simplifier2(data, length):
    """
    第一个数是总均值，第二个数是总方差
    此后每length返回两个数：length均值-总均值，max(abs(length值-length均值))
    所有数据均乘1000
    :param data:
    :param length:
    :return:
    """
    col_num = data.shape[1]
    answer = np.zeros((col_num, 72000 * 2 // length + 2))
    for i in range(col_num):
        all_mean = np.nanmean(data[:, i])
        if str(all_mean) == 'NaN' or str(all_mean) == 'nan':
            all_mean = all_std = 0
        else:
            all_std = np.nanstd(data[:, i])
        new_column = np.zeros(72000 * 2 // length + 2)
        new_column[0] = all_mean * 1000
        new_column[1] = all_std * 1000
        temp = []
        for j in range(72001):
            if j % length == 0 and j > 0:
                cur_mean = np.nanmean(temp)
                if str(cur_mean) == 'NaN' or str(cur_mean) == 'nan':
                    new_column[2 * j // length] = new_column[2 * j // length + 1] = 0
                else:
                    new_column[2 * j // length] = (cur_mean - all_mean) * 1000
                    new_column[2 * j // length + 1] = np.nanmax(np.abs(temp - cur_mean)) * 1000

                if j < 72000:
                    temp = [data[j][i]]
            else:
                temp.append(data[j][i])

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
        data = simplifier1(data, length)

    return data


def getLabel():
    return hdf5storage.loadmat('../label_r.mat')['manual_r']


def getDayLabel(num):
    wholeLabel = np.zeros((1, 912), dtype=np.int)
    label = getLabel()
    for j in range(24):
        for k in range(38):
            wholeLabel[0][j * 38 + k] = label[24 * num + j, k]
    return wholeLabel


def getDataLimited(num, simplify=True):
    """
    每种类型获取同样数量num的样本

    """


    data = np.zeros((72000, num * (7 - len(exclude))))
    wholeLabel = np.zeros((1, num * (7 - len(exclude))), dtype=np.int)
    label = getLabel()
    count = np.zeros(7)
    for i in range(len(exclude)):
        count[exclude[i] - 1] = 999

    col_index = 0
    finished = False
    for day in range(30):
        date = '2012-01-' + str(day + 1).zfill(2)
        if finished: break
        for hour in range(24):
            if finished: break
            datatemp = hdf5storage.loadmat('../' + date + '/' + date + ' ' + str(hour).zfill(2) + '-VIB.mat')['data']
            for j in range(38):
                cur_label = int(label[24 * day + hour, j])
                if count[cur_label - 1] >= num:
                    continue
                data[:, col_index] = datatemp[:, j]
                wholeLabel[0][col_index] = cur_label
                count[cur_label - 1] = count[cur_label - 1] + 1
                col_index = col_index + 1
                if min(count) >= num:
                    print('max day:{}'.format(day))  # 最后用到哪一天的数据
                    finished = True
                    break
    if simplify:
        data = simplifier1(data, length)

    return data, wholeLabel


def getDayLimited(daynum, simplify=True):
    data = getDayData(daynum, simplify)
    label = getDayLabel(daynum)
    newdata = []
    newlabel = []
    for i in range(len(label)):
        if label[i] not in exclude:
            newdata.append(data[:, i])
            newlabel.append(label[i])
    return newdata, newlabel


if __name__ == '__main__':
    data = getDayData(20, True)
    label = getDayLabel(20)
    # data,label = getDataLimited(100)
    print(data)
    print(label)
    print(data.shape)
    print(label.shape)
    count = np.zeros(7)
    for x in label:
        for y in x:
            count[int(y) - 1] = count[int(y) - 1] + 1
    print(count)
    """
    # plot
    for test_id in np.random.choice(range(140), 5):
        # for test_id in [38,67]:
        x = np.linspace(1, 3600, 72000 // length)
        y = []
        for rows in data:
            y.append(rows[test_id])
        plt.plot(x, y)
        print('当前类型{}'.format(label[0][test_id]))
        plt.show()
    """
