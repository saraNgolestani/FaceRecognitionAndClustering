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

