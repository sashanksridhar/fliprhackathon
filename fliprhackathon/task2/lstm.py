import csv
import numpy
from keras.layers import LSTM

from keras.models import Sequential
from keras.layers import Dense

Xi = [] # input train array
Xo = [] # output test array
Y = [] # input train value to be predicted

with open("E:\\fliprhackathon\\task2\\train.csv", 'r') as r,open("E:\\fliprhackathon\\task2\\test.csv", 'r') as r1:
    reader = csv.reader(r)
    testreader = csv.reader(r1)
    for row in reader:
        for j in range(0,len(row)):
            row[j] = int(row[j])
        Xi.append(row)
    for row in testreader:
        for j in range(0,len(row)):
            row[j] = int(row[j])
        Xo.append(row)


X = []  # input features

for i in Xi:
    X.append(i[:len(i)-1])  # input features extracted from train list
    Y.append(i[len(i)-1:][0])   # input value to be predicted

X = numpy.array(X)  # convert train list into array

X = X.reshape((X.shape[0], X.shape[1], 1))  # reshape from [samples, timesteps] into [samples, timesteps, features]

model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(3, 1)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, Y, epochs=100)

Xtest = numpy.array(Xo)
Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], 1))
predicted = model.predict(Xtest)
with open("E:\\fliprhackathon\\arima\\task2.csv", 'w') as w:
    writer = csv.writer(w, lineterminator='\n')
    writer.writerow(['people_ID', 'infect_prob'])
    j = 1
    for i in predicted:
            r = []
            r.append(j)
            r.append(i[0])
            writer.writerow(r)
            j+=1


