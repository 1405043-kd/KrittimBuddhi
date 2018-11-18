import pandas as pd

data = pd.read_fwf('E:/43/Pattern_off/Train_binary.txt', sep=" ", header=None)

weights = [-0.1, 0.20653640140000007, -0.23418117710000003, -0.2]

#print(data.iloc[0])

def predict(row, weight):
    activation = 0
    for i in range(len(row)-1):
        activation += weight[i] * row[i]
        print (weight[i], row[i])
    activation += weight[len(weight)-1]

    print(activation)

    if activation>=1:
        print ('Yes')
    else:
        print("No")
	#return 1.0 if activation >= 0.0 else 0.0

for index, row in data.iterrows():
    prediction = predict(row, weights)
