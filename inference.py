from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import pickle
from utils import *
import joblib
from matplotlib import pyplot
from argparse import ArgumentParser
import torch

from facenet_pytorch import MTCNN
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#crops the faces
mtcnn = MTCNN(margin=40, device=device, post_process=False)

# load tfl model
tfl_file = "facenet.tflite"
model = load_tflite_model(tfl_file)
print('Loaded Model')

# normalize input vectors
in_encoder = Normalizer(norm='l2')
out_encoder = LabelEncoder()
file = open("class.obj",'rb')
out_encoder = pickle.load(file)
file.close()
filename = 'classifer_model.sav'
# load the model from disk
model_svc = joblib.load(filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path")

    opt = parser.parse_args()
    # convert each face in the train set to an embedding
    img = read_image(opt.path)
    face = pre_process(mtcnn, img)
    embedding = predict(model, face)
    embedding = asarray([embedding])
    embedding = in_encoder.transform(embedding)

    yhat_class = model_svc.predict(embedding)
    yhat_prob = model_svc.predict_proba(embedding)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    if class_probability < 6:
        predict_names = 'unknown'

    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    # plot
    pyplot.imshow(img)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()