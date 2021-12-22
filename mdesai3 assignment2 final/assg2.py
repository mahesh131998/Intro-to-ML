from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import os, os.path
from keras.applications.xception import Xception
import numpy as np
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import pandas 
from pandas import DataFrame as df
from sklearn.metrics import silhouette_samples, silhouette_score
from validclust import dunn
from sklearn.metrics import pairwise_distances
import keras
from keras import layers
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import cv2

    

j=0
(x_train, _), (x_test, _) = cifar10.load_data()

#model
feature_list = []
model = Xception(weights='imagenet', include_top=False)
for i in range(len(x_test)) :
        im2 = x_test[i]
        img = preprocess_input(np.expand_dims(im2, axis=0))
        feature = model.predict(img)
        feature_np = np.array(feature)
        feature_list.append(feature_np.flatten())
        # print('Hi')
        print(j)
        j=j+1
        if j==8000:
            break



x = np.array(feature_list)
nos_classes=10
iterations=100
random_centroids = np.random.choice(len(x), nos_classes, replace=False)
centroids = x[random_centroids, :] 
distances = cdist(x, centroids ,'euclidean') 

points = np.array([np.argmin(i) for i in distances]) 

for _ in range(1,iterations): 
    centroids = []
    for idx in range(nos_classes):
        temp_cent = x[points==idx].mean(axis=0) 
        centroids.append(temp_cent)
 
    centroids = np.vstack(centroids) 
         
    distances = cdist(x, centroids ,'euclidean')
    points = np.array([np.argmin(i) for i in distances])    


sample_silhouette_values = silhouette_samples(feature_list, points)
print(sample_silhouette_values)
m=silhouette_score(feature_list, points)
print("silhoutte score for the cluster is =" , m)



dist = pairwise_distances(feature_list)
h=dunn(dist, points)
print("the DUNN score is =" ,h)




#AutoEncoder



from keras.datasets import cifar10
import numpy as np
from keras.callbacks import TensorBoard
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.metrics import  silhouette_score
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans

input_image = keras.Input(shape=(32, 32, 3))

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_image)
x = layers.MaxPooling2D((4, 4), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)



x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((4, 4))(x)
x = layers.Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)


decoded = layers.Conv2D(3,(3, 3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = cifar10.load_data()

x_train=x_train/255
print(x_test.shape)
print(x_train.shape)
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=300,
                shuffle=True,
                )

decoded_images = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_images[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



encoder = Model(autoencoder.input, autoencoder.layers[-4].output)
encoder.summary()
encoded_imgs = encoder.predict(x_test[:8000])
encoded_imgs= encoded_imgs.astype(int)
reshaped_data = encoded_imgs.reshape(len(encoded_imgs),-1)
x = np.array(reshaped_data)
nos_classes=10
iterations=100
random_centroids = np.random.choice(len(x), nos_classes, replace=False)
centroids = x[random_centroids, :] 
distances = cdist(x, centroids ,'euclidean') 
points = np.array([np.argmin(i) for i in distances]) 
for _ in range(iterations): 
    centroids = []
    for idx in range(nos_classes):
        temp_cent = x[points==idx].mean(axis=0) 
        centroids.append(temp_cent)
        
 
    centroids = np.vstack(centroids)       
    distances = cdist(x, centroids ,'euclidean')
    points = np.array([np.argmin(i) for i in distances]) 

reshaped_data = encoded_imgs.reshape(len(encoded_imgs),-1)
clusters = KMeans(10, random_state = 40)
h=clusters.fit(reshaped_data)
test_data = x_test.reshape(len(x_test),-1)

sco=silhouette_score( reshaped_data[:8000],points[:8000], metric='euclidean')
print("silhoutte score for the cluster is =" , sco)

