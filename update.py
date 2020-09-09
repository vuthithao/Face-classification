from utils import *
from mtcnn.mtcnn import MTCNN

#crops the faces
mtcnn = MTCNN()

# load train dataset
trainX, trainy = load_dataset(mtcnn, 'data/update/')
print(trainX.shape, trainy.shape)

# load tfl model
tfl_file = "models/facenet.tflite"
model = load_tflite_model(tfl_file)
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
  embedding = predict(model, face_pixels)
  newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

trainX_old = np.load('data/train_embs.npy')
trainy_old = np.load('data/train_y.npy')

newTrainX = np.concatenate((trainX_old, newTrainX), axis=0)
trainy = np.concatenate((trainy_old, trainy), axis=0)

np.save("data/train_embs.npy", newTrainX)
np.save("data/train_y.npy", trainy)