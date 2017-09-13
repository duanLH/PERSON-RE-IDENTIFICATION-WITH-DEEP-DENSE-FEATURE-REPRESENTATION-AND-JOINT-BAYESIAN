#coding=utf-8
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *
from sklearn.metrics.pairwise import *
import os.path as osp
from cmc import *
import os
from argparse import ArgumentParser

def excute_train(data_dir):

    train_data = osp.join(data_dir, 'train_features.npy')
    train_label = osp.join(data_dir,'train_labels.npy')
    result_fold ='./JointBayesian_model/'
    if not os.path.exists(result_fold):
        os.makedirs(result_fold) 
    data  = np.load(train_data)
    label = np.load(train_label)
    print data.shape, label.shape


    pca = PCA_Train(data, result_fold)
    data_pca = pca.transform(data)

    data_to_pkl(data_pca, result_fold+"pca_wdref.pkl")
    JointBayesian_Train(data_pca, label, result_fold)


def pair_distace(X,Y,A,G):
    X, Y = check_pairwise_arrays(X, Y)
    n_x, n_y = X.shape[0], Y.shape[0]
    D = np.zeros((n_x, n_y), dtype='float')
    for i in range(n_x):
        start = 0
        if X is Y:
            start = i
        for j in range(start, n_y):
            # distance assumed to be symmetric.
            D[i][j] = -(Verify(A, G,X[i], Y[j]))
            if X is Y:
                D[j][i] = D[i][j]
    return D

def _get_test_data(result_dir):
    PX = np.load(osp.join(result_dir, 'test_probe_features.npy'))
    PY = np.load(osp.join(result_dir, 'test_probe_labels.npy'))
    GX = np.load(osp.join(result_dir, 'test_gallery_features.npy'))
    GY = np.load(osp.join(result_dir, 'test_gallery_labels.npy'))
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(np.r_[PY, GY])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    PY = np.asarray([labels_map[l] for l in PY])
    GY = np.asarray([labels_map[l] for l in GY])
    return PX, PY, GX, GY
def excute_test(data_dir):
    result_fold = './JointBayesian_model/'
    PX, PY, GX, GY = _get_test_data(data_dir)
    with open(result_fold+"A.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
        G = pickle.load(f)


    clt_pca = joblib.load(result_fold + "pca_model.m")
    data_px = clt_pca.transform(PX)
    data_to_pkl(data_px, result_fold + "pca_px.pkl")

    PX = read_pkl(result_fold + "pca_px.pkl")


    data_gx = clt_pca.transform(GX)
    data_to_pkl(data_gx, result_fold + "pca_gx.pkl")

    GX = read_pkl(result_fold + "pca_gx.pkl")
    D=pair_distace(GX,PX,A,G)

    C = cmc(D, GY, PY)

    print('JointBayesian')
    for topk in [1, 5, 10, 20]:
        print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])


def main(args):
    data_dir = args.result_dir
    excute_train(data_dir)
    excute_test(data_dir)
if __name__ == '__main__':
    parser = ArgumentParser(
            description="Joint Bayesian")
    parser.add_argument('result_dir',
            help="Root directory of the Market-1501 dataset containing image files")
    args = parser.parse_args()
    main(args)
