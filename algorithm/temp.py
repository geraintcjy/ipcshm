import getData
import numpy as np


data = getData.getDayData(5, True)
print(data.shape)
# print(data[:, 911])
# print(np.nanmean(data[:, 911]))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!numpynb
