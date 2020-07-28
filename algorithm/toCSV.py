import getData
import csv

with open('../input/training_data.csv', encoding='utf-8', mode='w', newline='') as file1:
    writer = csv.writer(file1)
    for i in range(1):
        print('Get Training Data: Doing day', str(i))
        dayData = getData.getDayData(i)
        writer.writerows(dayData)

with open('../input/training_labels.csv', encoding='utf-8', mode='w', newline='') as file2:
    writer = csv.writer(file2)
    for i in range(1):
        print('Get Training Label: Doing day', str(i))
        dayLabel = getData.getDayLabel(i)
        writer.writerows(dayLabel)
'''
with open('../input/test_data.csv', encoding='utf-8', mode='w', newline='') as file3:
    writer = csv.writer(file3)
    for i in [1]:
        print('Get Test Data: Doing day', str(i))
        dayData = getData.getDayData(i)
        writer.writerows(dayData)

with open('../input/test_labels.csv', encoding='utf-8', mode='w', newline='') as file4:
    writer = csv.writer(file4)
    for i in [1]:
        print('Get Test Label: Doing day', str(i))
        dayLabel = getData.getDayLabel(i)
        writer.writerows(dayLabel)
'''