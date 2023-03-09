import numpy as np
from architect import *

data = np.load('Dataset.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

facenet_model = InceptionResNetV2()
weight_path = "./keras/weight/facenet_keras_weights.h5"

facenet_model.load_weights(weight_path)
print("Loaded model!")

def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean) / std
    sample = np.expand_dims(face, axis=0)
    y_pred = model.predict(sample)
    return y_pred[0]

emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)
    
emdTrainX = np.asarray(emdTrainX)

emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

np.savez_compressed('faces_embeddings.npz', emdTrainX, trainY, emdTestX, testY)