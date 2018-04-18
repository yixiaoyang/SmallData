from math import log
def calcEntropy(datas):
    entries = len(datas)
    labels = {}
    entropy = 0.0
    for data in datas:
        label = data[-1]
        if label not in labels.keys():
            labels[label] = 1
        else:
            labels[label] += 1
    for label in labels:
        prob = float(labels[label])/entries
        entropy -= prob*log(prob, 2)
    return entropy

array1=[
    [1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,0,'no'],
    [0,1,'no'],
]
array2=[
    [1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,0,'no'],
    [0,1,'maybe'],
]

print calcEntropy(array1)
print calcEntropy(array2)
