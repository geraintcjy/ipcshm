import getData
import csv

with open('training_data.csv', encoding='utf-8', mode='w', newline='') as file1:
    writer = csv.writer(file1)
    for i in range(1):
        print('Training Training Data: Doing day', str(i))
        dayData = getData.getDayData(i)
        writer.writerows(dayData)

with open('training_labels.csv', encoding='utf-8', mode='w', newline='') as file2:
    writer = csv.writer(file2)
    for i in range(1):
        print('Training Training Label: Doing day', str(i))
        dayLabel = getData.getDayLabel(i)
        writer.writerows(dayLabel)

with open('test_data.csv', encoding='utf-8', mode='w', newline='') as file3:
    writer = csv.writer(file3)
    for i in [1]:
        print('Training Test Data: Doing day', str(i))
        dayData = getData.getDayData(i)
        writer.writerows(dayData)

with open('test_labels.csv', encoding='utf-8', mode='w', newline='') as file4:
    writer = csv.writer(file4)
    for i in [1]:
        print('Training Test Label: Doing day', str(i))
        dayLabel = getData.getDayLabel(i)
        writer.writerows(dayLabel)
