import cv2
import numpy as np
from PIL import Image
from architect import *
from mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean) / std
    sample = np.expand_dims(face, axis=0)
    y_pred = model.predict(sample)
    return y_pred[0]

def detect_face(image):
    bounding_boxes = detector.detect_faces(image)
    return bounding_boxes

def draw_bounding_box(image, bboxes):
  for box in bboxes:
    x1, y1, w, h = box['box']
    x1, y1 = abs(x1), abs(y1)
    cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)

# Load model
facenet_model = InceptionResNetV2()
weight_path = "./keras/weight/facenet_keras_weights.h5"

facenet_model.load_weights(weight_path)

def predict_face(img_path):
  # Load dataset
  data = np.load('faces_embeddings.npz')
  emdTrainX, trainY, emdTestX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

  in_encoder = Normalizer()
  emdTrainX_norm = in_encoder.transform(emdTrainX)
  # emdTestX_norm = in_encoder.transform(emdTestX)

  out_encoder = LabelEncoder()
  out_encoder.fit(trainY)
  trainY_enc = out_encoder.transform(trainY)
  # testY_enc = out_encoder.transform(testY)

  model = SVC(kernel='linear', probability=True)
  model.fit(emdTrainX_norm, trainY_enc)
  print("Loaded model!")

  # Load face to predict
  face_box = extract_face(img_path)
  img_emb = get_embedding(facenet_model, face_box)

  img_expand = np.expand_dims(img_emb, axis=0)
  y_predict = model.predict(img_expand)
  y_prob = model.predict_proba(img_expand)

  class_index = y_predict[0]
  proba_predict = y_prob[0,class_index] * 100
  predict_names = out_encoder.inverse_transform(y_predict)

  print('Predicted: %s (%.3f)' % (predict_names[0], proba_predict))
  title = '%s (%.3f)' % (predict_names[0], proba_predict)
  return title