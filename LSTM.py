import pandas as pd
import numpy as np
from matplotlib import pyplot
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

print(np.shape(X_input),np.shape(Y_input))

from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed, Bidirectional, Dropout
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=False, random_state=0)
cvscores = []
i=0
print("Bidirectional LSTM with only recurrent dropout and dropout layer")


for j in range(11):

    X_inp = np.delete(X_input,j,axis=1)
    print("Feature %d is left out" %(j))

    for train, test in kfold.split(X_input, Y_input):
        X_train,Y_train = X_inp[train], Y_input[train]
        X_test, Y_test = X_inp[test], Y_input[test]

        X_train= (np.reshape(X_train,(90,112,10)))
        Y_train= (np.reshape(Y_train,(90,112,1)))

        X_test= (np.reshape(X_test,(10,112,10)))
        Y_test= (np.reshape(Y_test,(10,112,1)))

        model = Sequential()

        model.add(BatchNormalization(input_shape=(112,10)))

        model.add(Bidirectional(LSTM(50,input_shape=(112,10),activation='tanh',return_sequences=True,recurrent_dropout=0.5),merge_mode='ave'))
        model.add(Dropout(0.6))
        model.add(TimeDistributed(Dense(1,activation='sigmoid')))

        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])
        history = model.fit(X_train, Y_train, epochs=100, verbose=0, validation_data=(X_test,Y_test))
        scores = model.evaluate(X_test, Y_test, verbose=0)

        print("Subject: %d %s: %.2f%%" % (i+1,model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        i = i+1
        '''
    pyplot.figure(i)
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    i+=1
    '''

#pyplot.show()

    print('%.2f%% (+/- %.2f%%)' % ((np.mean(cvscores)), np.std(cvscores)))
