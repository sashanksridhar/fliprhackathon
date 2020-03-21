import csv
with open("E:\\fliprhackathon\\task2\\timeseries.csv", 'r', encoding='utf8') as r:
    X = []  # train features
    header = []  # training labels
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            header.extend(row)  # get labels alon
            c += 1
            continue
        x = []
        for i in range(0,len(row)):
            row[i] = int(row[i].replace(',','')) # as values are in the form of 1,202 etc convert into 1202 etc and then into int
        X.append(row[1:])  # get all features except people_id

    header.remove('people_ID')  # remove label as input features wont have this

    with open("E:\\fliprhackathon\\task2\\train.csv", 'w') as w,open("E:\\fliprhackathon\\task2\\test.csv", 'w') as w2:
        writer = csv.writer(w, lineterminator='\n') # for the train file
        writer1 = csv.writer(w2, lineterminator='\n')   # for the test file
# convert time series into series with intervals of 3 and the 4th value as output
        for i in X:
            t = []
            for j in range(0,5):
                r = []
                if(j==4): # this forms the final 3 values with the 4th value which is to be predicted and forms the test file
                    t.append(i[j])
                    t.append(i[j + 1])
                    t.append(i[j + 2])
                    writer1.writerow(t)
                else:
                    r.append(i[j])
                    r.append(i[j+1])
                    r.append(i[j+2])
                    r.append(i[j+3])

                    writer.writerow(r)

