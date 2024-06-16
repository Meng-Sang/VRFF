import cv2
import numpy as np


def consistency_estimation(mkp0, mkp1, method=cv2.LMEDS):
    if method == cv2.RANSAC:
        _, prediction = cv2.findHomography(mkp0, mkp1, method, ransacReprojThreshold=2.5, confidence=0.99999,
                                           maxIters=10000)
    else:
        _, prediction = cv2.findHomography(mkp0, mkp1, method, confidence=0.99999)
    prediction = np.array(prediction, dtype=bool).reshape([-1])
    mkp0 = mkp0[prediction]
    mkp1 = mkp1[prediction]
    return mkp0, mkp1


class IPF:
    def __init__(self, max_cap):
        self.max_cap = max_cap
        if self.max_cap < 1:
            self.max_cap = 1
        self.mkp_list = []

    def __call__(self, mkp_s):
        self.mkp_list.append(mkp_s)
        if len(self.mkp_list) > self.max_cap:
            self.mkp_list.pop(0)
        return np.concatenate(self.mkp_list, axis=1)


class Momentum:
    def __init__(self, alpha=0.99):
        self.v = None
        self.alpha = alpha

    def __call__(self, h):
        if self.v is None:
            self.v = h
        else:
            self.v = self.v * self.alpha + h * (1 - self.alpha)
        return self.v


class Deformation:
    def __init__(self, max_cap=47, fps=1, alpha=0.99, is_select_best=False):
        self.is_select_best = is_select_best
        self.candidate_estimation = []
        self.candidate_list = []
        self.step = 0
        self.ipf = IPF(max_cap)
        self.fps = int(fps)
        if alpha is not None:
            self.moment = Momentum(alpha)
        else:
            self.moment = None

    def get_homo_metrix(self, mkp0, mkp1):
        if mkp0.shape[0] < 4:
            return None
        h, prediction = cv2.findHomography(mkp0, mkp1, cv2.LMEDS, confidence=0.99999)
        if self.moment is not None:
            if not self.is_select_best or self.step > self.fps:
                h = self.moment(h)
            else:
                self.candidate_estimation.append(h)
        return h

    def select_best_estimation(self, val):
        if self.step > self.fps:
            return
        self.candidate_list.append(val.detach().cpu().numpy())
        if self.fps == self.step:
            index = np.argmax(np.array(self.candidate_list))
            self.moment.v = self.candidate_estimation[index]

    @staticmethod
    def get_tps_metrix(mkp0, mkp1):
        tps = cv2.createThinPlateSplineShapeTransformer()
        kp0 = mkp0.reshape(1, -1, 2)
        kp1 = mkp1.reshape(1, -1, 2)
        matches = []
        for j in range(1, mkp0.shape[0] + 1):
            matches.append(cv2.DMatch(j, j, 0))
        tps.estimateTransformation(kp1, kp0, matches)
        return tps

    def __call__(self, mkp0, mkp1, method="homo"):
        self.step += 1
        mkp0, mkp1 = consistency_estimation(mkp0, mkp1, cv2.RANSAC)
        mkp_pair = self.ipf([mkp0, mkp1])
        if method == "homo":
            return self.get_homo_metrix(mkp_pair[0], mkp_pair[1])
        elif method == "tps":
            return self.get_tps_metrix(mkp_pair[0], mkp_pair[1])
        else:
            raise Exception(f"unkown {method} method")
