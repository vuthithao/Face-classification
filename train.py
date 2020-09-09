from utils import *
from mtcnn.mtcnn import MTCNN

#crops the faces
mtcnn = MTCNN()

# load train dataset
trainX, trainy = load_dataset(mtcnn, 'data/train/')
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
np.save("data/train_embs.npy", newTrainX)
np.save("data/train_y.npy", trainy)