import pandas as pd

data = pd.read_fwf('E:/43/Pattern_off/Train_binary.txt', sep=" ", header=None)

weights = [0, 0, 0, 0]

#print(data.iloc[0])

def predict(row, weight):
    activation = 0
    for i in range(len(row)-1):
        activation += weight[i] * row[i]
        #print (weight[i], row[i])
    activation += weight[len(weight)-1]

    #print("activation: ", activation)

    if activation>=0:
        return 2
    else:
        return 1
	#return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, weights, l_rate, iterations):
    #weights = [0.0 for i in range(len(train[0]))]

    for itrn in range(iterations):
        sum_error = 0.0
        for index, row in train.iterrows():

            prediction = predict(row, weights)
            error = row[len(row)-1] - prediction
            sum_error += error**2
            #print(weights[3], "weights prev")
            weights[len(weights)-1] = weights[len(weights)-1] + l_rate * error
            #print(weights[3], "weights new")
            for i in range(len(row)-1):
                #print(i, "rowLen")
                weights[i] = weights[i] + l_rate * error * row[i]
        print(weights)
    return weights


train_weights(data, weights, .1, 10)

data = pd.read_fwf('E:/43/Pattern_off/Test_binary.txt', sep=" ", header=None)

#print(weights)
for index, row in data.iterrows():
    prediction = predict(row, weights)
    #if prediction==True:
     #   preRes=1
    #else:
     #   preRes=1

    print ("Out: ",prediction,  row[3]==prediction)
