import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

data = np.load('faces_embeddings.npz')
emdTrainX, trainY, emdTestX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print("Dataset: train= %d, test= %d" % (emdTrainX.shape[0], emdTestX.shape[0]))

in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY_enc = out_encoder.transform(trainY)
testY_enc = out_encoder.transform(testY)

model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainY_enc)

y_pred_train = model.predict(emdTrainX_norm)
y_pred_test = model.predict(emdTestX_norm)

score_train = accuracy_score(trainY_enc, y_pred_train)
score_test = accuracy_score(testY_enc, y_pred_test)

print('Accuracy: train= %.3f, test= %.3f' % (score_train, score_test))

print("\nAccuracy of train dataset: ")
print(classification_report(trainY_enc, y_pred_train))

print("\nAccuracy of test dataset: ")
print(classification_report(testY_enc, y_pred_test))