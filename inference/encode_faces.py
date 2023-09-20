from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument('-i', "--dataset", default="../data/testfaces", required=False, help="path to the directory of images")
ap.add_argument("-e", "--encodings", default="../output/encoded/encoded.pickle", required=False, help="output path for the serialized encoded faces")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="type of the face detection method to use")
args = ap.parse_args()

imagesPaths = list(paths.list_images(args.dataset))
data = []

for i, imagePath in enumerate(imagesPaths):

    print("[INFO] processing image {}/{}".format(i + 1, len(imagesPaths)))
    print(imagePath)
    image = cv2.imread(imagePath)

    boxes = face_recognition.face_locations(image, model=args.detection_method)
    encodings = face_recognition.face_encodings(image, boxes)

    d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

    print("[INFO] serializing encodings...")
    f = open(args.encodings, "wb")
    f.write(pickle.dumps(data))
    f.close()

