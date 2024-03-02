#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import dataset
dataset=pd.read_csv("Housing2.csv")
x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values
#encoding categorical data
#encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
columns_to_encode = [4, 5, 6, 7, 8, 10]  # Adjust these indices according to your dataset
for column in columns_to_encode:
    x[:, column] = le.fit_transform(x[:, column])
ct=ColumnTransformer(transformers=[('encoder5',OneHotEncoder(),[11])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
x=x.astype('float32')
y=y.astype('float32')
#splitting dataset in training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x_train[:,3]=sc.fit_transform((x_train[:,3]).reshape(-1,1)).flatten()
x_test[:,3]=sc.transform((x_test[:,3]).reshape(-1,1)).flatten()
y_train=sc.fit_transform(y_train.reshape(-1,1))
y_test=sc.transform(y_test.reshape(-1,1))
#build a model
#create a model
tf.random.set_seed(42)
model=tf.keras.Sequential([tf.keras.layers.Dense(100,activation='relu',input_shape=[14]),
                            tf.keras.layers.Dense(10,activation='relu'),
                            tf.keras.layers.Dense(1,activation='linear')])
#compile model
model.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                metrics=['mse'])
model_history=model.fit(x_train,y_train,epochs=30,validation_data=[x_test,y_test])
#predict values
y_pred=model.predict(x_test)
y_pred=y_pred.reshape(-1,1)
y_pred=sc.inverse_transform(y_pred)
#print(y_pred)
y_test=sc.inverse_transform(y_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#visualize model results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values')
plt.plot(y_pred, label='Predicted Values', alpha=0.7)
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample index')
plt.ylabel('Value of y')
plt.legend()
plt.show()