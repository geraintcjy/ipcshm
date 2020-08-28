import pandas
import numpy as np

da = pandas.DataFrame(np.random.randn(6, 4))
da = da.T
print(da)
se = [0, 0, 1, 1]
print(da.iloc[:, [1, 2, 3]])

if 1 in se:
    print(da.shape[1])
