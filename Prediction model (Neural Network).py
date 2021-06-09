import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = input('Enter Dataset location')


df = pd.read_csv(dataset)

X = df[['RPM','TPS (%)']].values 
y = df['Inj Q Tor (mg-st)'].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#NN
model = Sequential()

model.add(Dense(4, activation='relu')) #layer 1
model.add(Dense(4, activation='relu')) #layer 2
model.add(Dense(4, activation='relu')) #layer 3
model.add(Dense(4, activation='relu')) #layer 4
model.add(Dense(4, activation='relu')) #layer 5

model.add(Dense(1)) #layer 6

model.compile(optimizer='rmsprop', loss='mse') 
model.fit(x=X_train,y=y_train,epochs=250) 

test_predictions = model.predict(X_test) 
test_predictions = pd.Series(test_predictions.reshape(171,))

pred_df = pd.DataFrame(y_test,columns=['Test True Y'])

pred_df = pd.concat([pred_df,test_predictions],axis=1)

pred_df.columns = ['Test True Y', 'Model Predictions']

########################################
#Predicting a single value

RPM = int(input('Enter RPM'))
TPS = int(input('ENter TPS(%)'))

New_Input = [[RPM,TPS]]
New_Input = scaler.transform(New_Input)

print(model.predict(New_Input))

#########################################

#saving the model

model.save('HiPER.h5')

##########################################

#Loading the model back:
'''

from tensorflow import keras
model = keras.models.load_model('HiPER.h5')

'''