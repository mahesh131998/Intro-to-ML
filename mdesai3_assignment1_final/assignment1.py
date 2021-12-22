#-----------------------------------------------------------PART 1---------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv( "diabetes.csv" )
print(df.shape)
feature = df.iloc[:,:-1].values #X
target = df.iloc[:,-1:].values  #Y


X_train, X_rem, y_train, y_rem = train_test_split(feature, target, train_size=0.6, random_state = 6)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state = 4)

learning_rate = 0.0001
iterations = 300000
      
#fit
row_m , coloumn_n = X_train.shape
w= np.zeros(coloumn_n)
b=0

for i in range( iterations ) :
    A = 1 / ( 1 + np.exp( - ( X_train.dot( w ) + b ) ) )
    temp = ( A - y_train.T )   #Y_train.T       
    temp = np.reshape( temp, row_m )        
    dW = np.dot( X_train.T, temp ) / row_m         
    db = np.sum( temp ) / row_m 
          
        # update weights    
    w = w - learning_rate * dW    
    b = b - learning_rate * db

#predict
Z = 1 / ( 1 + np.exp( - ( X_test.dot( w ) + b ) ) )        
Y_p = np.where( Z > 0.5, 1, 0 )

correct_decided=0

counters=0
for counter in range( np.size( Y_p ) ) :  
        
        if y_test[counter] == Y_p[counter] :    #Y_train.T        
            correct_decided = correct_decided + 1
              
        counters = counters + 1

print( "Accuracy on test set by our model       :  ", ( correct_decided / counters ) * 100 )

#-------------------------------------------------------------------PART 1 ------ ENDS-----------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------PART 2---------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf

df = pd.read_csv( "diabetes.csv" )
feature = df.iloc[:,:-1].values #X
target = df.iloc[:,-1:].values  #Y

X_train, X_rem, y_train, y_rem = train_test_split(feature, target, train_size=0.6, random_state = 6)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state = 4)


np.random.seed(22)
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(X_train.shape[1],),kernel_regularizer=tf.keras.regularizers.l1()))
model.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],),kernel_regularizer=tf.keras.regularizers.l1()))  # Hidden layer.-- loss = l1 * reduce_sum(abs(x))
model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],),kernel_regularizer=tf.keras.regularizers.l1()))  # Hidden layer. 
model.add(Dense(1, activation='sigmoid'))  # Output layer.(Since there are 2 classes)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
grp = model.fit(X_train, y_train, batch_size=32, epochs=1000, verbose=1, validation_data=(X_valid, y_valid),shuffle=False)
loss, accuracy = model.evaluate(X_test,y_test, verbose=1)
print("Loss : "+str(loss))
print("Accuracy :"+str(accuracy*100.0))


# Accuracy
plt.figure(figsize=(15, 10))
plt.plot(grp.history['accuracy'])
plt.plot(grp.history['val_accuracy'])
plt.title(' Training and Testing Error for L1 or L2')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower right')
plt.savefig('Accuracy_BS_32_IT_1000.jpg')
plt.show()


# Loss
plt.figure(figsize=(15, 10))
plt.plot(grp.history['loss'])
plt.plot(grp.history['val_loss'])
plt.title('Training and Testing Loss forL1 or L2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='lower right')
plt.savefig('Loss_BS_32_IT_1000.jpg')
plt.show()

#-------------------------------------------------------------------------------PART 2-------------------ENDS----------------------------------------------------------------------




#-----------------------------------------------------------------------------------PART 3------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf

df = pd.read_csv( "diabetes.csv" )
feature = df.iloc[:,:-1].values #X
target = df.iloc[:,-1:].values  #Y

X_train, X_rem, y_train, y_rem = train_test_split(feature, target, train_size=0.6, random_state = 6)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state = 4)

np.random.seed(22)
model = Sequential()

model.add(Dense(4, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],)))  
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))  # Output layer.(Since there are 2 classes)  

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
grp = model.fit(X_train, y_train, batch_size=50, epochs=1500, verbose=1, validation_data=(X_valid, y_valid),shuffle=False)
loss, accuracy = model.evaluate(X_test,y_test, verbose=2)
print("Loss : "+str(loss))
print("Accuracy :"+str(accuracy*100.0))


# Accuracy
plt.figure(figsize=(15, 10))
plt.plot(grp.history['accuracy'])
plt.plot(grp.history['val_accuracy'])
plt.title(' Training and Testing Error for Dropout')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper right')
plt.savefig('Accuracy.jpg')
plt.show()

# Loss
plt.figure(figsize=(15, 10))
plt.plot(grp.history['loss'])
plt.plot(grp.history['val_loss'])
plt.title('Training and Testing Loss for Dropout')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
plt.savefig('Loss.jpg')
plt.show()


#-----------------------------------------------------------------PART 3----------------------------ENDS-------------------------------------------------------------------------   
