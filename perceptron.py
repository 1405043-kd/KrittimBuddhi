import pandas as pd

weights_1f = [0.1, 0.1, 0.1, 0.1]
weights_2f = [0.1, 0.1, 0.1]
weights_4f = [0.1, 0.1, 0.1, 0.1, 0.1]

# print(data.iloc[0])

def predict(row, weight):
    activation = 0
    for i in range(len(row) - 1):
        activation += weight[i] * row[i]
        # print (weight[i], row[i])
    activation += weight[len(weight) - 1]

    # print("activation: ", activation)

    if activation > 0:
        return 2
    else:
        return 1
    # return 1.0 if activation >= 0.0 else 0.0


def trainer_basic(train, weights, l_rate, iterations):
    # weights = [0.0 for i in range(len(train[0]))]
    for itrn in range(iterations):
        mis_classifieds = {}
        mis_classified_no = 0
        sumt = 0
        for index, row in train.iterrows():
            # rowlen=len(row)

            prediction = predict(row, weights)
            error = row[len(row) - 1] - prediction
            #print(error)
            if error != 0:
                mis_classifieds[mis_classified_no] = row
                mis_classifieds[mis_classified_no]['class_e'] = error
                mis_classified_no += 1

            # for i in range(len(row) - 1):
            # weights[i] = weights[i] + l_rate*error*row[i]
            # sumt += error * row[i]

        for row_miss in mis_classifieds:
            weights[len(weights) - 1] = weights[len(weights) - 1] + l_rate * mis_classifieds[row_miss]['class_e']
            for i in range(len(mis_classifieds[row_miss]) - 1):
                weights[i] = weights[i] + l_rate * mis_classifieds[row_miss][i] * mis_classifieds[row_miss]['class_e']

        print(weights)
    return weights


def trainer_r_p(train, weights, l_rate, iterations):
    for itrn in range(iterations):

        for index, row in train.iterrows():

            prediction = predict(row, weights)
            error = row[len(row) - 1] - prediction

            weights[len(weights) - 1] = weights[len(weights) - 1] + l_rate * error

            for i in range(len(row) - 1):
                weights[i] = weights[i] + l_rate * error * row[i]
        print(weights)
    return weights


def trainer_pocket(train, weights, l_rate, iterations):
    pocket = weights
    min_error = 300

    for itrn in range(iterations):

        for index, row in train.iterrows():

            prediction = predict(row, weights)
            error = row[len(row) - 1] - prediction

            # print(weights[3], "weights prev")
            weights[len(weights) - 1] = weights[len(weights) - 1] + l_rate * error
            # print(weights[3], "weights new")
            for i in range(len(row) - 1):
                # print(i, "rowLen")
                weights[i] = weights[i] + l_rate * error * row[i]
            error_now = len(predictor_on_dataset(train, weights))
            if error_now <= min_error:
                pocket = weights
                min_error = error_now
        print(weights)
        # print(pocket,"errors: ", min_error, error_now)
    return pocket


def predictor_on_dataset(data, weights):
    falseCount = 0
    mis_classified = {}
    for index, row in data.iterrows():
        prediction = predict(row, weights)
        # if prediction==True:
        #   preRes=1
        # else:
        #   preRes=1
        # print ("Out: ",prediction,  row[len(row)-1]==prediction)
        if row[len(row) - 1] != prediction:
            mis_classified[falseCount] = row
            falseCount += 1
    return mis_classified


###############################kesler#######################################
def add_dict(d1, d2):
    d3 = {}
    for i in d1:
        d3[i] = d1[i] + d2[i]
    return d3


def sub_dict(d1, d2):
    d3 = {}
    for i in d1:
        d3[i] = d1[i] - d2[i]
    return d3


def conf_cal(item, weight):
    confidence = 0;
    for k in weight:
        confidence += weight[k] * item[k];
    return confidence;


def kesler_train(data, iterations):
    features = {}
    n_feature = len(data.columns)
    for i in range(n_feature - 1):
        features[i] = data[i]
    # print(features[2])

    n_class = len(data[n_feature - 1].unique())
    # print(n_class)
    #
    weights = {}
    for c in range(n_class):
        weights[c] = {}
        for f in range(n_feature - 1):
            weights[c][f] = 0
    # print(weights)
    # print()

    for itrt in range(iterations):
        for index, row in data.iterrows():
            row_c = row[:-1]
            # print(row_c)

            y = -1
            yy = 0

            for c in range(n_class):
                confidence = conf_cal(row_c, weights[c]);
                # print(confidence)
                if confidence > y:
                    y = confidence
                    yy = c

            if row[n_feature - 1] != (yy + 1):
                weights[yy] = sub_dict(weights[yy], row_c);
                weights[row[n_feature - 1] - 1] = add_dict(weights[row[n_feature - 1] - 1], row_c);

    return weights


def kesler_predict(data, kesler_weights):
    mis_classified = {}
    mis_count = 0
    n_feature = len(data.columns)
    n_class = len(data[n_feature - 1].unique())

    for index, row in data.iterrows():
        row_c = row[:-1]
        # print(row_c)

        y = -1
        yy = 0
        for c in range(n_class):
            confidence = conf_cal(row_c, kesler_weights[c]);
            # print(confidence)
            if confidence > y:
                y = confidence
                yy = c + 1
        print("prediction: ", yy, " label_actual: ", row[n_feature - 1], yy == row[n_feature - 1])
        if yy != row[n_feature - 1]:
            mis_classified[mis_count] = row
            mis_count += 1
    return mis_classified

#
# # -------------------------kesler----------------
#
# data = pd.read_fwf('F:/43/Pattern_off/eval/kesler/Train.txt', sep="  ", header=None)
# kesler_weights = kesler_train(data, 100)
# data = pd.read_fwf('F:/43/Pattern_off/eval/kesler/Test.txt', sep="  ", header=None)
# mis_classified_kesler=(kesler_predict(data, kesler_weights))
# print("mis_classifieds: ", (mis_classified_kesler))
# print("mis_classifieds num: ", len(mis_classified_kesler))
#
#
# # -------------------------kesler----------------


# print (data)


# -------------------------First_3_Variants----------------
#



#data = pd.read_fwf('F:/43/Pattern_off/eval/perceptron/trainLinearlySeparable.txt', delimiter="  ", header=None)

with open ('F:/43/Pattern_off/eval/perceptron/trainLinearlySeparable.txt') as f:
#with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
    data = []
    for line in f:
        #print(line)
        data.append([float(x) for x in line.split()])

data= pd.DataFrame(data)
#print(data)
#weights = trainer_basic(data, weights_4f, 1, 1000)

weights = trainer_r_p(data, weights_4f, 1, 10)

#weights = trainer_pocket(data, weights_4f, .1, 10)

#data = pd.read_fwf('F:/43/Pattern_off/eval/perceptron/testLinearlySeparable.txt', sep=" ", header=None)
with open ('F:/43/Pattern_off/eval/perceptron/testLinearlySeparable.txt') as f:
#with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
    data = []
    for line in f:
        #print(line)
        data.append([float(x) for x in line.split()])

data= pd.DataFrame(data)

mis_classified = predictor_on_dataset(data, weights)
print("miss_classified ", mis_classified)
print(len(mis_classified))


# -------------------------First_3_Variants----------------
