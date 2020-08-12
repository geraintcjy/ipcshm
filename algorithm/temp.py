import csv

with open('../input/training_labels.csv', encoding='utf-8', mode='r', newline='') as file1:
    reader = csv.reader(file1)
    for rows in reader:
        labels = rows

print(labels)
for i in range(len(labels)):
    if str(labels[i]) == '4':
        labels[i] = 1
    else:
        labels[i] = 0
print(labels)

with open('../input/training_labels4.csv', encoding='utf-8', mode='w', newline='') as file2:
    writer = csv.writer(file2)
    writer.writerow(labels)
