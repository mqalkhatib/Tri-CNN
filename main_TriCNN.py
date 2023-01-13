'''
Code implementation of "Tri-CNN: A Three Branch Model for Hyperspectral Image Classification"

Mohammed Q. Alkhatib (mqalkhatib@ieee.org)    
'''

from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import numpy as np
import scipy
from utils import *
from model_TriCNN import Tri_CNN



## GLOBAL VARIABLES
dataset = 'GP'
windowSize = 9
PCA_comp = 45

X, y = loadData(dataset)

X.shape, y.shape

# Apply PCA for dimensionality reduction
X,pca = applyPCA(X,PCA_comp)


X, y = createImageCubes(X, y, windowSize=windowSize)


Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, 0.99)

ytrain = keras.utils.to_categorical(ytrain)

Xtrain = np.expand_dims(Xtrain, axis=4)
Xtest = np.expand_dims(Xtest, axis=4)


# Early stopper: if the model stopped improving for 10 consecutive epochs, 
# stop the training
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )

model = Tri_CNN(Xtrain, num_classes(dataset))
model.summary()
    
history = model.fit(Xtrain, ytrain,
                    batch_size = 16, 
                    verbose=1, 
                    epochs=100, 
                    shuffle=True, 
                    callbacks = [early_stopper])
    
    
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)
    
    
kappa = cohen_kappa_score(ytest,  y_pred_test)
oa = accuracy_score(ytest, y_pred_test)
confusion = confusion_matrix(ytest, y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)
    
       
 

print('OverallAccuracy  = ', format(oa*100, ".2f") + ' %')
print('Average Accuracy = ', format(aa*100, ".2f") + ' %')
print('Kappa            = ', format(kappa*100, ".2f") + ' %')



######################### To create the predicted class image #################

# load the original image
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
X,pca = applyPCA(X, numComponents=PCA_comp)
X = padWithZeros(X, windowSize//2)

# calculate the predicted image, this is a pixel wise operation, will take long time
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if i%25 == 0 and j%25 ==0: 
            print("i = " + str(i) + ", j = " + str(j))
        if target == 0 :
            continue
        else :
            image_patch=Patch(X,i,j, windowSize)
            X_test_image = image_patch.reshape(1, image_patch.shape[0],
                                               image_patch.shape[1], 
                                               image_patch.shape[2], 1).astype('float32')                                   
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction+1
 

# Save the output
scipy.io.savemat('Tri_CNN_' + dataset +'.mat', {'outputs': outputs})

