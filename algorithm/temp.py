exclude = [1, 2, 4]
whole = [1, 2, 3, 4, 5, 6, 7]
k = 0
label_map = {}

for item in whole:
    if item not in exclude:
        label_map[item] = k
        k += 1

print(label_map)
