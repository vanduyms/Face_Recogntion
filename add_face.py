import cv2
import numpy as np
from PIL import Image
from architect import *
from mtcnn import MTCNN
from keras.models import load_model

# Load model
facenet_model = InceptionResNetV2()
weight_path = "./keras/weight/facenet_keras_weights.h5"

facenet_model.load_weights(weight_path)

detector = MTCNN()

# Load the dataset
dataset = np.load('faces_embedding.npz')

def add_face(img, name):
  img = cv2.imread(img)
  name = np.asarray(name)

  result = detector.detect_faces(img)
  x, y, h, w = result[0]['box']

  face = img[y:y+h, x:x+w]

  image = Image.fromarray(face)
  image = image.resize((160,160))
  face_array = np.asarray(image)

  face = face_array.astype('float32')
  mean, std = face.mean(), face.std()
  face = (face-mean) / std
  sample = np.expand_dims(face, axis=0)
  embedding = facenet_model.predict(sample)[0]

  data = np.concatenate((data['arr_0'], embedding), axis=0)
  data_names = np.concatenate((data['arr_1'], name), axis=0)

  new_dataset_file = 'faces_embedding.npz'
  np.savez(new_dataset_file, embeddings=data, names=data_names)