from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default='../output/encoded/encoded.pickle', help='Path to the encoded pickle file')
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of parallel jobs to run (-1 will use all CPUs)")
args = ap.parse_args()

print("[INFO] loading encodings ...")
data = pickle.loads(open(args.encodings, "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

print("[INFO] clustering ...")
cluster = DBSCAN(metric="euclidean", n_jobs=args.jobs)
cluster.fit(encodings)

labelIDs = np.unique(cluster.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

for labelID in labelIDs:
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(cluster.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(15, len(idxs)),
                            replace=False)
    faces = []

    for i in idxs:
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        face = cv2.resize(face, (96, 96))
        faces.append(face)

        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        montage = build_montages(faces, (96, 96), (5, 5))[0]

        # show the output montage
        title = "Face ID #{}".format(labelID)
        title = "Unknown Faces" if labelID == -1 else title
        cv2.imshow(title, montage)
        # cv2.waitKey(0)
