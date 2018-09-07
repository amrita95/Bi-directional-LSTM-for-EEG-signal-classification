import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed, Bidirectional, Dropout,Reshape
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold,GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


#Data Preprocessing
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
features2 = df.drop(col,axis=1)
aa = np.array(features2)
X_input= aa[:,0:11]
Y_input =  aa[:,11]

print(np.shape(X_input))
def create_model(neurons=1):
    model = Sequential()
    model.add(Reshape((112, 11), input_shape=(1232,)))

    model.add(BatchNormalization(input_shape=(112,11)))

    model.add(Bidirectional(LSTM(50,input_shape=(112,11),activation='tanh',return_sequences=True,recurrent_dropout=0.5),merge_mode='ave'))
    model.add(Dropout(0.6))
    model.add(TimeDistributed(Dense(1,activation='sigmoid')))
    model.add(Reshape((112,)))
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))
    return model


#Building keras model
kfold = KFold(n_splits=10, shuffle=False, random_state=0)
cvscores = []
i=0
print("Bidirectional LSTM with only recurrent dropout and dropout layer")

model = KerasClassifier(build_fn=create_model, verbose=0)

epochs = [50, 100, 150, 200]
batches = [5, 10, 20, 50, 100]
param_grid = dict(epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold,  scoring='accuracy',verbose=50)
X = np.reshape(X_input, (100, 1232))
Y = np.reshape(Y_input, (100, 112))
print("Actual input: {}".format(X.shape))
print("Actual output: {}".format(Y.shape))

grid_result = grid.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

