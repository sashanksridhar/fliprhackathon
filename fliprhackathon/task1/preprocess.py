from sklearn import preprocessing
import pandas as pd
import csv

with open("E:\\fliprhackathon\\task1\\Train_dataset.csv", 'r', encoding='utf8') as r, open(
        "E:\\fliprhackathon\\task1\\Test_dataset.csv", 'r', encoding='utf8') as t:
    X = []  # train features
    Y = []  # train output probability
    header = []  # training labels
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            header.extend(row)  # get labels alon
            c += 1
            continue
        X.append(row[:len(row) - 1])  # get all features except probability
        Y.append(float(row[len(row) - 1]))  # probability
    Xtest = []  # test features
    headertest = []  # test labels
    c = 0
    reader = csv.reader(t)
    for row in reader:
        if c == 0:
            headertest.extend(row)
            c += 1
            continue
        Xtest.append(row)

    header.remove('Infect_Prob')  # remove output label as input features wont have this

    df = pd.DataFrame(X, columns=header) # training data frame
    dftest = pd.DataFrame(Xtest, columns=headertest) # testing dataframe

    df = df.drop(columns=['people_ID', 'Designation', 'Name'])  # remove columns that are not needed
    dftest = dftest.drop(columns=['people_ID', 'Designation', 'Name'])

    header.remove('people_ID') # remove column name from label list
    header.remove('Designation')
    header.remove('Name')

    headertest.remove('people_ID')
    headertest.remove('Designation')
    headertest.remove('Name')




    frames = [df,dftest]
    combine = pd.concat(frames, ignore_index=True) # merge both train and test data frames. Will be useful to calc max of each column

    region = preprocessing.LabelEncoder() # create encoder
    region.fit(combine['Region'])   # fit with all values from train and test set for that column
    df['Region'] = region.transform(df['Region'])   # encode train values
    dftest['Region'] = region.transform(dftest['Region'])   # encode test values
    max_reg = max(df['Region'].max(), dftest['Region'].max())    # calc max between train and test set for that column
    if max_reg!=0:
        df['Region'] = df['Region'].div(float(max_reg)) # normalize train set
        dftest['Region'] = dftest['Region'].div(float(max_reg)) # normalize test set

    gender = preprocessing.LabelEncoder()
    gender.fit(combine['Gender'])
    df['Gender'] = gender.transform(df['Gender'])
    dftest['Gender'] = gender.transform(dftest['Gender'])
    max_gen = max(df['Gender'].max(), dftest['Gender'].max())
    if max_gen != 0:
        df['Gender'] = df['Gender'].div(float(max_gen))
        dftest['Gender'] = dftest['Gender'].div(float(max_gen))

    marry = preprocessing.LabelEncoder()
    marry.fit(combine['Married'])
    df['Married'] = marry.transform(df['Married'])
    dftest['Married'] = marry.transform(dftest['Married'])
    max_mar = max(df['Married'].max(), dftest['Married'].max())
    if max_mar != 0:
        df['Married'] = df['Married'].div(float(max_mar))
        dftest['Married'] = dftest['Married'].div(float(max_mar))

    df['Children'] = df['Children'].apply(pd.to_numeric) # convert string value read from csv to numeric value
    dftest['Children'] = dftest['Children'].apply(pd.to_numeric)
    max_ch = max(df['Children'].max(), dftest['Children'].max())    # calc max between train and test set for that column
    if max_ch!=0:
        df['Children'] = df['Children'].div(float(max_ch)) # normalize train set
        dftest['Children'] = dftest['Children'].div(float(max_ch)) # normalize test set

    occ = preprocessing.LabelEncoder()
    occ.fit(combine['Occupation'])
    df['Occupation'] = occ.transform(df['Occupation'])
    dftest['Occupation'] = occ.transform(dftest['Occupation'])
    max_occ = max(df['Occupation'].max(), dftest['Occupation'].max())
    if max_occ != 0:
        df['Occupation'] = df['Occupation'].div(float(max_occ))
        dftest['Occupation'] = dftest['Occupation'].div(float(max_occ))

    trans = preprocessing.LabelEncoder()
    trans.fit(combine['Mode_transport'])
    df['Mode_transport'] = trans.transform(df['Mode_transport'])
    dftest['Mode_transport'] = trans.transform(dftest['Mode_transport'])
    max_trans = max(df['Mode_transport'].max(), dftest['Mode_transport'].max())
    if max_trans != 0:
        df['Mode_transport'] = df['Mode_transport'].div(float(max_trans))
        dftest['Mode_transport'] = dftest['Mode_transport'].div(float(max_trans))

    df['cases/1M'] = df['cases/1M'].apply(pd.to_numeric)
    dftest['cases/1M'] = dftest['cases/1M'].apply(pd.to_numeric)
    max_cases = max(df['cases/1M'].max(), dftest['cases/1M'].max())
    if max_cases!=0:
        df['cases/1M'] = df['cases/1M'].div(float(max_cases))
        dftest['cases/1M'] = dftest['cases/1M'].div(float(max_cases))

    df['Deaths/1M'] = df['Deaths/1M'].apply(pd.to_numeric)
    dftest['Deaths/1M'] = dftest['Deaths/1M'].apply(pd.to_numeric)
    max_deaths = max(df['Deaths/1M'].max(), dftest['Deaths/1M'].max())
    if max_deaths!=0:
        df['Deaths/1M'] = df['Deaths/1M'].div(float(max_deaths))
        dftest['Deaths/1M'] = dftest['Deaths/1M'].div(float(max_deaths))

    comorbidity = preprocessing.LabelEncoder()
    comorbidity.fit(combine['comorbidity'])
    df['comorbidity'] = comorbidity.transform(df['comorbidity'])
    dftest['comorbidity'] = comorbidity.transform(dftest['comorbidity'])
    max_com = max(df['comorbidity'].max(), dftest['comorbidity'].max())
    if max_com != 0:
        df['comorbidity'] = df['comorbidity'].div(float(max_com))
        dftest['comorbidity'] = dftest['comorbidity'].div(float(max_com))

    df['Age'] = df['Age'].apply(pd.to_numeric)
    dftest['Age'] = dftest['Age'].apply(pd.to_numeric)
    max_age = max(df['Age'].max(), dftest['Age'].max())
    if max_age!=0:
        df['Age'] = df['Age'].div(float(max_age))
        dftest['Age'] = dftest['Age'].div(float(max_age))

    df['Coma score'] = df['Coma score'].apply(pd.to_numeric)
    dftest['Coma score'] = dftest['Coma score'].apply(pd.to_numeric)
    max_coma = max(df['Coma score'].max(), dftest['Coma score'].max())
    if max_coma!=0:
        df['Coma score'] = df['Coma score'].div(float(max_coma))
        dftest['Coma score'] = dftest['Coma score'].div(float(max_coma))

    pul = preprocessing.LabelEncoder()
    pul.fit(combine['Pulmonary score'])
    df['Pulmonary score'] = pul.transform(df['Pulmonary score'])
    dftest['Pulmonary score'] = pul.transform(dftest['Pulmonary score'])
    max_pul = max(df['Pulmonary score'].max(), dftest['Pulmonary score'].max())
    if max_pul != 0:
        df['Pulmonary score'] = df['Pulmonary score'].div(float(max_pul))
        dftest['Pulmonary score'] = dftest['Pulmonary score'].div(float(max_pul))

    card = preprocessing.LabelEncoder()
    card.fit(combine['cardiological pressure'])
    df['cardiological pressure'] = card.transform(df['cardiological pressure'])
    dftest['cardiological pressure'] = card.transform(dftest['cardiological pressure'])
    max_card = max(df['cardiological pressure'].max(), dftest['cardiological pressure'].max())
    if max_card != 0:
        df['cardiological pressure'] = df['cardiological pressure'].div(float(max_card))
        dftest['cardiological pressure'] = dftest['cardiological pressure'].div(float(max_card))

    df['Diuresis'] = df['Diuresis'].apply(pd.to_numeric)
    dftest['Diuresis'] = dftest['Diuresis'].apply(pd.to_numeric)
    max_di = max(df['Diuresis'].max(), dftest['Diuresis'].max())
    if max_di!=0:
        df['Diuresis'] = df['Diuresis'].div(float(max_di))
        dftest['Diuresis'] = dftest['Diuresis'].div(float(max_di))

    df['Platelets'] = df['Platelets'].apply(pd.to_numeric)
    dftest['Platelets'] = dftest['Platelets'].apply(pd.to_numeric)
    max_pl = max(df['Platelets'].max(), dftest['Platelets'].max())
    if max_pl!=0:
        df['Platelets'] = df['Platelets'].div(float(max_pl))
        dftest['Platelets'] = dftest['Platelets'].div(float(max_pl))

    df['HBB'] = df['HBB'].apply(pd.to_numeric)
    dftest['HBB'] = dftest['HBB'].apply(pd.to_numeric)
    max_hbb = max(df['HBB'].max(), dftest['HBB'].max())
    if max_hbb!=0:
        df['HBB'] = df['HBB'].div(float(max_hbb))
        dftest['HBB'] = dftest['HBB'].div(float(max_hbb))

    df['d-dimer'] = df['d-dimer'].apply(pd.to_numeric)
    dftest['d-dimer'] = dftest['d-dimer'].apply(pd.to_numeric)
    max_ddimer = max(df['d-dimer'].max(), dftest['d-dimer'].max())
    if max_ddimer!=0:
        df['d-dimer'] = df['d-dimer'].div(float(max_ddimer))
        dftest['d-dimer'] = dftest['d-dimer'].div(float(max_ddimer))

    df['Heart rate'] = df['Heart rate'].apply(pd.to_numeric)
    dftest['Heart rate'] = dftest['Heart rate'].apply(pd.to_numeric)
    max_hr = max(df['Heart rate'].max(), dftest['Heart rate'].max())
    if max_hr!=0:
        df['Heart rate'] = df['Heart rate'].div(float(max_hr))
        dftest['Heart rate'] = dftest['Heart rate'].div(float(max_hr))

    df['HDL cholesterol'] = df['HDL cholesterol'].apply(pd.to_numeric)
    dftest['HDL cholesterol'] = dftest['HDL cholesterol'].apply(pd.to_numeric)
    max_hdl = max(df['HDL cholesterol'].max(), dftest['HDL cholesterol'].max())
    if max_hdl!=0:
        df['HDL cholesterol'] = df['HDL cholesterol'].div(float(max_hdl))
        dftest['HDL cholesterol'] = dftest['HDL cholesterol'].div(float(max_hdl))

    df['Charlson Index'] = df['Charlson Index'].apply(pd.to_numeric)
    dftest['Charlson Index'] = dftest['Charlson Index'].apply(pd.to_numeric)
    max_ci = max(df['Charlson Index'].max(), dftest['Charlson Index'].max())
    if max_ci!=0:
        df['Charlson Index'] = df['Charlson Index'].div(float(max_ci))
        dftest['Charlson Index'] = dftest['Charlson Index'].div(float(max_ci))

    df['Blood Glucose'] = df['Blood Glucose'].apply(pd.to_numeric)
    dftest['Blood Glucose'] = dftest['Blood Glucose'].apply(pd.to_numeric)
    max_glu = max(df['Blood Glucose'].max(), dftest['Blood Glucose'].max())
    if max_glu!=0:
        df['Blood Glucose'] = df['Blood Glucose'].div(float(max_glu))
        dftest['Blood Glucose'] = dftest['Blood Glucose'].div(float(max_glu))

    df['Insurance'] = df['Insurance'].apply(pd.to_numeric)
    dftest['Insurance'] = dftest['Insurance'].apply(pd.to_numeric)
    max_ins = max(df['Insurance'].max(), dftest['Insurance'].max())
    if max_ins!=0:
        df['Insurance'] = df['Insurance'].div(float(max_ins))
        dftest['Insurance'] = dftest['Insurance'].div(float(max_ins))

    df['salary'] = df['salary'].apply(pd.to_numeric)
    dftest['salary'] = dftest['salary'].apply(pd.to_numeric)
    max_sal = max(df['salary'].max(), dftest['salary'].max())
    if max_sal!=0:
        df['salary'] = df['salary'].div(float(max_sal))
        dftest['salary'] = dftest['salary'].div(float(max_sal))

    df['FT/month'] = df['FT/month'].apply(pd.to_numeric)
    dftest['FT/month'] = dftest['FT/month'].apply(pd.to_numeric)
    max_ft = max(df['FT/month'].max(), dftest['FT/month'].max())
    if max_ft!=0:
        df['FT/month'] = df['FT/month'].div(float(max_ft))
        dftest['FT/month'] = dftest['FT/month'].div(float(max_ft))

    df['prob'] = Y

    df.dropna(inplace=True)

    l = df.values.tolist() # train dataframe to list


    with open("E:\\fliprhackathon\\task1\\normal_train.csv", 'w') as w:
        writer = csv.writer(w, lineterminator='\n')
        writer.writerow(header)
        for i in range(0, len(l)):
            writer.writerow(l[i])

    l = dftest.values.tolist()  # test dataframe to list

    with open("E:\\fliprhackathon\\task1\\normal_test.csv", 'w') as w:
        writer = csv.writer(w, lineterminator='\n')
        writer.writerow(headertest)
        for i in range(0, len(l)):
            writer.writerow(l[i])
