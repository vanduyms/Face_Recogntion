import os
import cv2
import mtcnn
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from mtcnn.mtcnn import MTCNN

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

def load_face(dir):
    faces = list()
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(dir):
    X, y = list(), list()

    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]

        print("loaded %d sample for class: %s" % (len(faces),subdir))
        
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

X, Y = load_dataset('./Data/')
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state=42)

np.savez_compressed('Dataset.npz', trainX, trainY, testX, testY)