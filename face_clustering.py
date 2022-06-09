from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np

for base, dirs, files in os.walk("/home/deeplab/Masaüstü/berkaydurur/face_clustering_test/pickles/"):
    for Files in files:
        print(Files)
        print(base)
        print("[INFO] loading encodings...")
        data = pickle.loads(open(base + Files, "rb").read())
        data = np.array(data)
        encodings = [d["encoding"] for d in data]
        # cluster the embeddings
        print("[INFO] clustering...")
        clt = DBSCAN(metric="euclidean", n_jobs=-1)
        clt.fit(encodings)
        # determine the total number of unique faces found in the dataset
        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] , unique faces: {}".format(numUniqueFaces))
        # loop over the unique face integers
