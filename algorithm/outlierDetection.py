import numpy as np
import matplotlib.pyplot as plt

    
def detect(sample,sd_time):
    '''
    异常点探测
    sample: 原始数据
    sd_time: 标准差倍数
    '''
    all_err=[]
    e_s = sample
    batch_size = 3
    window_size = 100
    #找到窗口数目
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
    #得到窗口e_s
    for j in range(1, num_windows + 2):
        prior_idx = (j - 1) * (batch_size)
        # 前面有i-1个batch size
        idx = (window_size * batch_size) + ((j - 1) * batch_size)
        if j == num_windows + 1:
            # 因为最后一个加的幅度不满于config.batchsize
            idx = len(sample)
        window_e_s = e_s[prior_idx:idx]
        # x = np.linspace(1, 3600, 4800)
        # y = []
        # print(prior_idx,idx)
        # for pt in e_s:
        #     y.append(pt)
        # plt.plot(x, y)
        # yy = []
        # for ii in window_e_s:
        #     yy.append(ii)
        # plt.plot(np.linspace(prior_idx/4800*3600,idx/4800*3600,int(idx-prior_idx)),yy,color='red')
        # plt.show()

        error_buffer = 3
        perc_high, perc_low = np.percentile(sample, [95, 5])
        mean = np.mean(window_e_s)
        sd = np.std(window_e_s)
        i_anom = []
        E_seq = []
        for x in range(0, len(window_e_s)):
            anom = True
            if window_e_s[x] < mean + sd_time*sd and window_e_s[x] > mean - sd_time*sd:
                anom = False
            if anom:
                # print(window_e_s[x],mean ,sd)
                for b in range(0, error_buffer):
                    if not prior_idx + x + b in i_anom and not prior_idx + x + b >= len(e_s) :
                            i_anom.append(prior_idx + x + b)
                    if not prior_idx + x - b in i_anom and not prior_idx + x - b < 0:
                            i_anom.append(prior_idx + x - b)
        all_err=np.append(all_err,i_anom)
        # groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
    x = np.linspace(1, 3600, len(e_s))
    y = []
    all_err = np.array(sorted(list(set(all_err))))
    err_count = 0
    for ii in range(len(all_err)-1):
        if all_err[ii]+1 in all_err:
            continue
        else:
            err_count += 1
    if err_count > 0.5: 
        err_count += 1
    else:
        for pt in e_s:
            y.append(pt)
        plt.plot(x, y)
        yy = []
        for ii in all_err:
            yy.append(e_s[int(ii)])
        plt.plot(all_err/len(e_s)*3600+1,yy,color='red')
        plt.show()
        return err_count
    # print(sd,err_count)

    return err_count
    #inter_range和chan_std如上所示
    #error_buffer是异常点周围被判定为异常区间的范围