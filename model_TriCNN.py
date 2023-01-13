from tensorflow.keras.layers import BatchNormalization, Conv3D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input
import tensorflow as tf


def Tri_CNN(img_list, num_class, lr=1e-3):
    input1=Input(shape=img_list.shape[1:])
       
    # Spectral Path        
    c1=Conv3D(64,(1,1,3),activation="relu")(input1)
    c1 = BatchNormalization()(c1)
    c1=Conv3D(64,(1,1,3),activation="relu")(c1)

    layer1=Flatten()(c1)
  
    
    # Spatial Path
    c2=Conv3D(64,(3,3,1),activation="relu")(input1)
    c2 = BatchNormalization()(c2)
    c2=Conv3D(64,(3,3,1),activation="relu")(c2)

    layer2=Flatten()(c2)
    
    
    # Spectral-Spatial Path
    c3=Conv3D(64,(3,3,3),activation="relu")(input1)
    c3 = BatchNormalization()(c3)
    c3=Conv3D(64,(3,3,3),activation="relu")(c3)

    layer3=Flatten()(c3)
    
    # Feature Fusion (Concatenation)
    result=tf.concat([layer1,layer2,layer3],axis=1);
    
    # Fully connected
    layer11=Dense(512,activation="relu")(result)
    layer11=Dropout(0.3)(layer11)
    layer11=Dense(256,"relu")(layer11)
    layer11=Dropout(0.3)(layer11)
    
    
    predict=Dense(num_class,activation="softmax")(layer11)
    
    model=Model(inputs=[input1],outputs=predict)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
    
    
