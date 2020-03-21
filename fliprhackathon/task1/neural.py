import csv
import numpy
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense

Xi = [] # input train array
Xo = [] # output test array
Y = []  # input train probabilities

with open("E:\\fliprhackathon\\task1\\normal_train.csv", 'r') as r,open("E:\\fliprhackathon\\task1\\normal_test.csv", 'r') as r1:
    c = 0
    reader = csv.reader(r)  # read train data set
    for row in reader:
        if c==0:
            c+=1
            continue
        for j in range(0,len(row)):
            row[j] = float(row[j])  # convert str values into float
        Xi.append(row)              # input array
    testreader = csv.reader(r1)     # read test data set
    c = 0
    for row in testreader:
        if c==0:
            c+=1
            continue
        for j in range(0,len(row)):
            row[j] = float(row[j]) # convert str values into float
        Xo.append(row)   # output array


X = []  # features to be trained
for i in Xi:
    X.append(i[:len(i)-1])  # extract features from input train array
    Y.append(i[len(i)-1:][0])   # extract train probability from input train array

model = Sequential()    # ANN
model.add(Dense(24, input_dim=24, activation='relu'))   # input layer
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1)) # output layer

optimizer = RMSprop(0.0099)
model.compile(loss='mean_squared_error', optimizer=optimizer) # compile model

Xout = numpy.array(X)   # convert training list into numpy array
model.fit(Xout, Y, epochs=750)  # fit model
Xtest = numpy.array(Xo) # convert testing list into numpy array
predicted = model.predict(Xtest) # test model

with open("E:\\fliprhackathon\\output_task1.csv", 'w') as w:
    writer = csv.writer(w, lineterminator='\n')
    writer.writerow(['people_ID', 'infect_prob'])
    j = 1
    for i in predicted:
        r = []
        r.append(j)
        r.append(i[0])
        writer.writerow(r)
        j += 1