from sklearn import neighbors, linear_model, svm, neural_network, ensemble, preprocessing, naive_bayes, metrics,tree
import numpy as np

data = []

Attributes = [
    ['?','Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    ['?','Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    ['?','Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    ['?','Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    ['?','Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    ['?','White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    ['?','Female', 'Male'],
    ['?','United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines'
        , 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
     'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
]
#All preprocessed
totalData = []
totalY = []

count = 0
with open("Data/train_final.csv") as file:

    for line in file:
        count += 1
        if count == 1:
            continue
        point = line.strip('\n').split(',')
        dataPoint3 = []

        dataPoint3.append(float(point[0]))
        #dataPoint3.append(float(point[0])**2)

        for i in range(len(Attributes[0])):
            if point[1] == Attributes[0][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)

        #dataPoint3.append(float(point[2]))

        for i in range(len(Attributes[1])):
            if point[3] == Attributes[1][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)

        dataPoint3.append(float(point[4]))

        # for i in range(len(Attributes[2])):
        #     if point[5] == Attributes[2][i]:
        #         dataPoint3.append(i)
        #         #dataPoint3.append(i**2)
        for i in range(len(Attributes[3])):
            if point[6] == Attributes[3][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)
        for i in range(len(Attributes[4])):
            if point[7] == Attributes[4][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)
        for i in range(len(Attributes[5])):
            if point[8] == Attributes[5][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)
        for i in range(len(Attributes[6])):
            if point[9] == Attributes[6][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)

        dataPoint3.append(float(point[10]))
        #dataPoint3.append(float(point[10])**2)

        dataPoint3.append(float(point[11]))
        #dataPoint3.append(float(point[11])**2)

        dataPoint3.append(float(point[12]))
        #dataPoint3.append(float(point[12])**2)


        for i in range(len(Attributes[7])):
            if point[13] == Attributes[7][i]:
                dataPoint3.append(i)
                #dataPoint3.append(i**2)

        totalData.append(dataPoint3)
        totalY.append(int(point[-1]))


#testData = trainingData[20000:]
#testY = trainingY[20000:]
#trainingData = trainingData[:20000]
#trainingY = trainingY[:20000]

testData = totalData[20000:]
testY = totalY[20000:]
trainingData = totalData[:20000]
trainingY = totalY[:20000]

'''
model = ensemble.ExtraTreesClassifier(criterion="entropy")
print("ExtraTrees")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))
'''

model = ensemble.AdaBoostClassifier(n_estimators=7500,learning_rate=1.7)
print("AdaBoost")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

'''
model = ensemble.BaggingClassifier()
print("Bagging")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = ensemble.GradientBoostingClassifier(n_estimators=100,loss="exponential",learning_rate=0.45)
print("GradientBoost")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = ensemble.RandomForestClassifier(n_estimators=100,max_samples=5000,criterion="entropy",max_features=2)
print("RandomForest")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)
print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = naive_bayes.MultinomialNB()
print("Multinomial")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = naive_bayes.ComplementNB()
print("Complement")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = naive_bayes.BernoulliNB()
print("Bernoulli")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = naive_bayes.GaussianNB()
print("Gaussian")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = svm.SVC()
print("SVC")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = linear_model.LogisticRegression(max_iter=10000)
print("LogisticRegression")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = neural_network.MLPClassifier([100,100,100],activation="logistic")
print("NeuralNetwork")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = neighbors.KNeighborsClassifier(n_neighbors=5)
print("NearestNeighbors")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))
'''


testX = []
ID = []
count = 0
with open("Data/test_final.csv") as file:
    for line in file:
        count += 1
        if count == 1:
            continue
        point = line.strip('\n').split(',')
        ID.append(point[0])
        dataPoint3 = []

        dataPoint3.append(float(point[1]))
        #dataPoint3.append(float(point[1]) ** 2)

        for i in range(len(Attributes[0])):
            if point[2] == Attributes[0][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)

        # dataPoint3.append(float(point[2]))

        for i in range(len(Attributes[1])):
            if point[4] == Attributes[1][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)

        dataPoint3.append(float(point[5]))

        # for i in range(len(Attributes[2])):
        #     if point[5] == Attributes[2][i]:
        #         dataPoint3.append(i)
        #         #dataPoint3.append(i**2)

        for i in range(len(Attributes[3])):
            if point[7] == Attributes[3][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)
        for i in range(len(Attributes[4])):
            if point[8] == Attributes[4][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)
        for i in range(len(Attributes[5])):
            if point[9] == Attributes[5][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)
        for i in range(len(Attributes[6])):
            if point[10] == Attributes[6][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)

        dataPoint3.append(float(point[11]))
        #dataPoint3.append(float(point[11]) ** 2)

        dataPoint3.append(float(point[12]))
        #dataPoint3.append(float(point[12]) ** 2)

        dataPoint3.append(float(point[13]))
        #dataPoint3.append(float(point[13]) ** 2)

        for i in range(len(Attributes[7])):
            if point[14] == Attributes[7][i]:
                dataPoint3.append(i)
                # dataPoint3.append(i**2)

        testX.append(dataPoint3)

testX = np.asarray(testX)

print("Testing")
with open("Submission.csv","w") as file:
    file.write("ID,Prediction\n")
    prediction = model.decision_function(testX)
    for i in range(len(testX)):
        file.write(ID[i] + "," + str(prediction[i]) + "\n")
