import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
import os
from PIL import Image, ImageFilter
import cv2

def compute_kmm(source_positive_dir_path,target_positive_dir_path):
    openDirectory = os.getcwd()
    # source_positive_dir_path = os.path.join(openDirectory, "datasets", "after_elite_sample", fruit_name, "source","train", "positive")
    # target_positive_dir_path = os.path.join(openDirectory, "datasets", "after_elite_sample", fruit_name, "target","train","positive")
    data_name_sp = os.listdir(source_positive_dir_path)
    data_name_tp = os.listdir(target_positive_dir_path)
    for j in range(len(data_name_sp)):
        path_now = os.path.join(source_positive_dir_path, data_name_sp[j])
        if j == 0:
            KMM_source_positive_sample = openimage(path_now).reshape((1, -1))
        elif j > 0:
            KMM_source_positive_sample = np.append(KMM_source_positive_sample, openimage(path_now).reshape((1, -1)),axis=0)
    for k in range(len(data_name_tp)):
        path_now = os.path.join(target_positive_dir_path, data_name_tp[k])
        if k == 0:
            KMM_target_positive_sample = openimage(path_now).reshape((1, -1))
        elif k > 0:
            KMM_target_positive_sample = np.append(KMM_target_positive_sample, openimage(path_now).reshape((1, -1)),axis=0)
    KMM_source_positive_sample, KMM_target_positive_sample = np.asarray(KMM_source_positive_sample), np.asarray(KMM_target_positive_sample)
    KMM_source_positive_sample = KMM_source_positive_sample.astype(np.double)
    KMM_target_positive_sample = KMM_target_positive_sample.astype(np.double)

    kmm = KMM(kernel_type='rbf',B=2)
    beta = kmm.fit(KMM_source_positive_sample, KMM_target_positive_sample)
    return beta[0]

def openimage(path):
    return (cv2.cvtColor(cv2.resize(cv2.imread(path),(224,224)),cv2.COLOR_BGR2RGB)).flatten()

def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            # Compute the linear kernel between X1 and X2, Gram matrix array of shape (X1_samples, X2_samples)
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            # Compute the rbf (gaussian) kernel between X1 and X2, Gram matrix array of shape (X1_samples, X2_samples)
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K

class KMM:
    def __init__(self, kernel_type='rbf', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel, like K(x, y) = exp(-gamma ||x-y||^2), If None, defaults to 1.0 / n_features
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta

