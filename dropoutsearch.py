import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed, Bidirectional, Dropout
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold

def createmodel(drop=0.5,rec_drop=0.5):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(112,11)))

    model.add(Bidirectional(LSTM(50,input_shape=(112,11),activation='tanh',return_sequences=True,recurrent_dropout=rec_drop),merge_mode='ave'))
    model.add(Dropout(drop))
    model.add(TimeDistributed(Dense(1,activation='sigmoid')))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])

    return model


csvfile = '/home/amrita95/Desktop/HBA Project/EEG_data.csv'
samples = pd.read_csv(csvfile)
col = ['SubjectID','VideoID','user-definedlabeln']
features = samples.drop(col,axis=1)

a = samples.groupby(['SubjectID','VideoID'])[features.columns[0]].count()

df = pd.DataFrame(columns=samples.columns)
for i in range(10):
    for j in range(10):
        a = samples.loc[(samples['SubjectID']==i) & (samples['VideoID']==j)]
        df = df.append(a[0:112])

col = ['SubjectID','VideoID','predefinedlabel']
features = df.drop(col,axis=1)
aa = np.array(features)



X_input= aa[:,0:11]
Y_input =  aa[:,11]


kfold = KFold(n_splits=10, shuffle=False, random_state=0)
cvscores = []
i=0
print("Bidirectional LSTM with only recurrent dropout and dropout layer")



drop = [0.3,0.4, 0.5, 0.6, 0.7]
rec_drop = [0.3,0.4, 0.5, 0.6, 0.7]

for e in drop:
    for b in rec_drop:
        for train, test in kfold.split(X_input, Y_input):
            X_train,Y_train = X_input[train], Y_input[train]
            X_test, Y_test = X_input[test], Y_input[test]
            X_train= (np.reshape(X_train,(90,112,11)))
            Y_train= (np.reshape(Y_train,(90,112,1)))

            X_test= (np.reshape(X_test,(10,112,11)))
            Y_test= (np.reshape(Y_test,(10,112,1)))
            model = createmodel(drop=e,rec_drop=b)
            history = model.fit(X_train, Y_train, epochs=50,batch_size=20, verbose=0, validation_data=(X_test,Y_test))
            scores = model.evaluate(X_test, Y_test, verbose=0)

            print("Subject: %d %s: %.2f%%" % (i+1,model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            i = i+1

        #pyplot.show()

        print('For drop:%f and rec_drop:%f %.2f%% (+/- %.2f%%)' % (e,b,np.mean(cvscores), np.std(cvscores)))
