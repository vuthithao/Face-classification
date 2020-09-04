# from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed

from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from utils import *
import torch

from facenet_pytorch import MTCNN
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#crops the faces
mtcnn = MTCNN(margin=40, device=device, post_process=False)

# load train dataset
trainX, trainy = load_dataset(mtcnn, 'data/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(mtcnn, 'data_cr/val/')
# save arrays to one file in compressed format
savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)

# load the face dataset
data = load('faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# load tfl model
tfl_file = "facenet.tflite"
model = load_tflite_model(tfl_file)
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
  embedding = predict(model, face_pixels)
  newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
  embedding = predict(model, face_pixels)
  newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

data = load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

filehandler = open("class.obj","wb")
pickle.dump(out_encoder,filehandler)
filehandler.close()

trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


# save the model to disk
filename = 'classifer_model.sav'
pickle.dump(model, open(filename, 'wb'))