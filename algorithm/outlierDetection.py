import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import getData


def showPlt(sample):
    x = np.linspace(1, 3600, len(sample))
    y = []
    for pt in sample:
        y.append(pt)
    y = np.array(y)
    plt.plot(x, y)
    plt.show()


def classification_6_7(sample, debug=False):
    model = linear_model.LinearRegression()
    y = []
    x = []
    for index, pt in enumerate(sample):
        if index > 0:
            if sample[index] == sample[index - 1] or abs(sample[index]) < 1e-7:
                continue
        y.append(pt)
        x.append(index*3600/(72000/getData.length))
    # x = np.linspace(1, 3600, len(y))
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    model.fit(x, y)
    score = model.score(x, y)
    if debug:
        print(score)
    if score > 0.95:
        return 6
    else:
        return 7


def detect(sample, sd_time):
    '''
    异常点探测
    sample: 原始数据
    sd_time: 标准差倍数
    '''
    all_err = []
    e_s = sample
    batch_size = 3
    window_size = 100
    # 找到窗口数目
    num_windows = int((len(sample) - (batch_size * window_size)) / batch_size)
    # decrease the historical error window size (h) if number of test values is limited
    while num_windows < 0:
        # 如果 windowsize过大 不断减少 找到刚好的windowsize
        window_size -= 1
        if window_size <= 0:
            window_size = 1
        num_windows = int((len(sample) - (batch_size * window_size)) / batch_size)
        # y_test长度小于batchsize
        if window_size == 1 and num_windows < 0:
            raise ValueError("Batch_size (%s) larger than test_data (len=%s). Adjust it." % (
                batch_size, len(sample)))
    # 得到窗口e_s
    for j in range(1, num_windows + 2):
        prior_idx = (j - 1) * (batch_size)
        # 前面有i-1个batch size
        idx = (window_size * batch_size) + ((j - 1) * batch_size)
        if j == num_windows + 1:
            # 因为最后一个加的幅度不满于config.batchsize
            idx = len(sample)
        window_e_s = e_s[prior_idx:idx]

        error_buffer = 3
        mean = np.mean(window_e_s)
        sd = np.std(window_e_s)
        i_anom = []
        for x in range(0, len(window_e_s)):
            anom = True
            if mean + sd_time * sd > window_e_s[x] > mean - sd_time * sd:
                anom = False
            if anom:
                for b in range(0, error_buffer):
                    if not prior_idx + x + b in i_anom and not prior_idx + x + b >= len(e_s):
                        i_anom.append(prior_idx + x + b)
                    if not prior_idx + x - b in i_anom and not prior_idx + x - b < 0:
                        i_anom.append(prior_idx + x - b)
        all_err = np.append(all_err, i_anom)

    all_err = np.array(sorted(list(set(all_err))))
    err_count = 0
    for ii in range(len(all_err) - 1):
        if all_err[ii] + 1 in all_err:
            continue
        else:
            err_count += 1
    if err_count > 0.5:
        err_count += 1
    return err_count
    # error_buffer是异常点周围被判定为异常区间的范围
