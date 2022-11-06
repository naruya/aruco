import cv2
import cv2.aruco as aruco
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import imageio


def calibrate(vid, aruco_dict, charuco):
    corners_all = []
    ids_all = []
    frames = []

    for t, img in enumerate(vid):
        img = deepcopy(img)
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=aruco_dict)

        if not (ids is not None and len(ids) > 1):
            print("skip (t={}, ids={})".format(t, ids))
            continue

        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco)

        if response < 8:
            print("skip (t={}, response={})".format(t, response))
            continue

        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
        frames.append(img)

    print("Number of detected frames:", len(corners_all))

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=charuco,
            imageSize=vid[0].shape[1::-1],
            cameraMatrix=None,
            distCoeffs=None)

    print('ret\n', ret, '\nmtx\n', cameraMatrix, '\ndist\n', distCoeffs)

    # for debug
    imageio.mimwrite("temp_calib{:04}.mp4".format(np.random.randint(1000)), frames, macro_block_size=8)

    return cameraMatrix, distCoeffs


def undistort(vid, mtx, dist):
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, vid[0].shape[1::-1], 0)
    _undistort = lambda img: cv2.undistort(deepcopy(img), mtx, dist, None, newmtx)
    return [_undistort(img) for img in tqdm(vid)]


def estimate_pose(vid, aruco_dict, charuco, mtx, dist, sl, mode='A4'):
    indices, vid_paint, rvec_hist, tvec_hist = [], [], [], []

    for t, img in enumerate(vid):
        img = deepcopy(img[...,::-1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_params = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = charuco,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = mtx,
                distCoeffs = dist)

        img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255), ids=ids)

        if not (ids is not None and len(ids) > 1):
            print("skip (t={}, ids={})".format(t, ids))
            continue

        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco)

        if response < 8:
            print("skip (t={}, response={})".format(t, response))
            continue

        # valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
        #         charucoCorners=charuco_corners,
        #         charucoIds=charuco_ids,
        #         board=charuco,
        #         cameraMatrix=mtx,
        #         distCoeffs=dist,
        #         rvec=None,
        #         tvec=None)

        objp = np.empty((0,3), np.float32)
        for idx in charuco_ids:
            if mode=='A4':
                # objpi = charuco.chessboardCorners[idx] - np.array([[-2,3.5,0]]) * sl
                # objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[-2,3.5,0]]) * sl, np.array([[0,1,0], [-1,0,0], [0,0,-1]]))
                # objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[-2,3.5,0]]) * sl, np.array([[1,0,0], [0,-1,0], [0,0,-1]]))
                objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[-2,3.5,0]]) * sl, np.array([[0,-1,0], [-1,0,0], [0,0,-1]]))
            elif mode=='A3':
                objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[5,7,0]]) * sl, np.array([[-1,0,0], [0,1,0], [0,0,-1]]))
            objp = np.append(objp, objpi, axis=0)

        valid, rvec, tvec = cv2.solvePnP(objp, charuco_corners, mtx, dist)

        if not valid:
            print("skip (t={}, valid={})".format(t, valid))
            continue

        img = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 0.3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        indices.append(t)
        vid_paint.append(img)
        rvec_hist.append(rvec)
        tvec_hist.append(tvec)

    print("Number of detected frames:", len(indices))

    # for debug
    imageio.mimwrite("temp_pose{:04}.mp4".format(np.random.randint(1000)), vid_paint, macro_block_size=8)

    return indices, rvec_hist, tvec_hist


class Warper():
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher()

    def __call__(self, float_img, ref_img):
        float_kp, float_des = self.akaze.detectAndCompute(float_img, None)
        ref_kp, ref_des = self.akaze.detectAndCompute(ref_img, None)

        matches = self.bf.knnMatch(float_des, ref_des, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # matches_img = cv2.drawMatchesKnn(
        #     float_img,
        #     float_kp,
        #     ref_img,
        #     ref_kp,
        #     good_matches,
        #     None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # plt.imshow(matches_img); plt.show()

        ref_matched_kpts = np.float32(
            [float_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32(
            [ref_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(
            ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

        warped_image = cv2.warpPerspective(
            float_img, H, (float_img.shape[1], float_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        # plt.imshow(warped_image); plt.show()

        return warped_image


import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from copy import deepcopy


class FeatureExtractor():
    def __init__(self):
        self.net = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(0).eval()
        self.feature_extractor = create_feature_extractor(self.net, {
            # "layer1.0.bn1": "conv1",
            # "layer1.0.bn2": "conv2",
            # "layer1.1.bn1": "conv3",
            "layer1.1.bn2": "conv4",
        })
    def __call__(self, x):
        x = torch.from_numpy(x.transpose(2,0,1)[None]).float().to(0)
        with torch.no_grad():
            features = self.feature_extractor(x)

        features = features.values()
        # features = [torch.sigmoid(x) for x in features]
        features = [x.squeeze().cpu().numpy().transpose(1,2,0) for x in features]
        features = np.concatenate(features, axis=2)
        return np.array(features)


class BackRemover():
    def __init__(self):
        self.warper = Warper()
        self.extractor = FeatureExtractor()

    def __call__(self, real, pred, upperlim=12., thresh=0.6, size=(960, 720)):
        orig = deepcopy(real)
        
        real = cv2.resize(real, size)
        pred = cv2.resize(pred, size)
        real = cv2.blur(real, (3,3))
        pred = cv2.blur(pred, (3,3))
        pred = self.warper(pred, real)

        real = self.extractor(real)
        pred = self.extractor(pred)

        diff = np.mean(np.abs(real - pred), axis=2)

        # print(diff.max(), np.percentile(diff, [98]))
        diff = np.clip(diff, 0, upperlim) / upperlim

        diff = np.where(diff < thresh, 0, diff)

        segm = orig / 255.
        diff = (diff.astype(np.float32) * 255).astype(np.uint8)
        diff = cv2.resize(diff, segm.shape[0:2][::-1])
        segm = segm.astype(np.float32) * diff.astype(np.float32)[...,None] / 255.
        segm = (segm * 255.).astype(np.uint8)
        return segm


def project_w2c(pts, rvec, tvec, mtx, dist):
    pts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
    pts = pts.squeeze().astype(np.int16)  # np.int16!!!
    return pts