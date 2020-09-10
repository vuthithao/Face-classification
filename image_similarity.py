from sklearn.preprocessing import Normalizer
from utils import *
from argparse import ArgumentParser

from mtcnn.mtcnn import MTCNN
from scipy.spatial import distance
import time
import os
import shutil


# convert each face in the train set to an embedding
def check(img, trainX, trainy):
    img2 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces, boxes = pre_process(mtcnn, img)
    embeddings = [predict(model, face) for face in faces]

    for i in range(len(embeddings)):
        distances = [distance.euclidean(embeddings[i], k) for k in trainX]
        class_probability = np.min(distances)
        predict_names = trainy[np.argmin(distances)]
        if class_probability > 0.6:
            predict_names = 'unknown'
        result = predict_names
        color = (0, 0, 255)
        cv2.putText(img2, result, (int(boxes[i][0] + 5), int(boxes[i][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return img2, predict_names

def checkAndGet(trainX=None, trainy=None, con=True):
    newTrainX = []
    if trainX is not None:
        cap = cv2.VideoCapture(0)
        xxx = time.time()
        checkk = []
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            img, predict_names = check(frame, trainX, trainy)
            checkk.append(predict_names)
            # Display the resulting frame
            cv2.imshow('check', img)
            unknown = True
            for a in checkk:
                if a != 'unknown':
                    unknown = False
            if time.time() - xxx > 5 and unknown:
                if predict_names == 'unknown':
                    break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        con = False
        cap.release()
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        xxx = time.time()
        while (True):
            ret, img = cap.read()
            img2 = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces, boxes = pre_process(mtcnn, img)

            for i in range(len(boxes)):
                result = 'unknown'
                color = (0, 0, 255)
                cv2.putText(img2, result, (int(boxes[i][0] + 5), int(boxes[i][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color, 1)
            cv2.imshow('check', img2)
            if time.time() - xxx > 5:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        print('Do you want be a part of us? (y/n)')
        x = input()
        if x == 'y':
            print('Enter your name:')
            y = input()
            os.mkdir('data/custom/' + y)

            for pose in range(len(poses)):
                start = time.time()
                cap = cv2.VideoCapture(0)
                while (True):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    # Display the resulting frame
                    img_show = frame.copy()
                    path = 'data/sample/' + poses[pose] + '.jpg'
                    im = cv2.imread(path)
                    img_show [:im.shape[0], :im.shape[1]] = im
                    cv2.imwrite(os.path.join('data/custom', y, poses[pose] + '.jpg'), frame)
                    cv2.imshow('xxx', img_show)
                    if time.time() - start > 3:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        con = False
                        break
                cap.release()
                cv2.destroyAllWindows()

            # load train dataset
            trainX, trainy = load_dataset(mtcnn, 'data/custom/')
            # convert each face in the train set to an embedding
            newTrainX = list()
            for face_pixels in trainX:
                embedding = predict(model, face_pixels)
                newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)

            np.save("data/custom/train_embs.npy", newTrainX)
            np.save("data/custom/train_y.npy", trainy)
        else:
            con = False
    return con, newTrainX, trainy

def run(trainX=None, trainy=None):
    con, trainX, trainy = checkAndGet(trainX, trainy)
    if con:
        run(trainX, trainy)
    else:
        print('goodbye!')


#crops the faces
mtcnn = MTCNN()

# load tfl model
tfl_file = "models/facenet.tflite"
model = load_tflite_model(tfl_file)
print('Loaded Model')

# normalize input vectors
in_encoder = Normalizer(norm='l2')
poses = ['front', '3_4_lelf', '3_4_right', 'from_bellow', 'from_above']


if __name__ == "__main__":

    print('Do you want reset?: (y/n)')
    reset = input()
    if reset == 'y':
        p = 'data/custom'
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.mkdir(p)
        run()
    else:
        try:
            trainX = np.load('data/custom/train_embs.npy')
            trainy = np.load('data/custom/train_y.npy')
            run(trainX, trainy)
        except:
            run()


