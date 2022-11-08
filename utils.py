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

        if response < 4:
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

        if response < 6:
            print("skip (t={}, response={})".format(t, response))
            continue

        # ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
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

        ret, rvec, tvec = cv2.solvePnP(objp, charuco_corners, mtx, dist)

        if not ret:
            print("skip (t={}, ret={})".format(t, ret))
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

        if len(good_matches) < 4:
            return False, "warp error! (len(good_maches) == {})".format(len(good_matches))

        # matches_img = cv2.drawMatchesKnn(
        #     float_img,
        #     float_kp,
        #     ref_img,
        #     ref_kp,
        #     good_matches,
        #     None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        ref_matched_kpts = np.float32(
            [float_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32(
            [ref_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(
            ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

        warped_image = cv2.warpPerspective(
            float_img, H, (float_img.shape[1], float_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        # plt.imshow(matches_img); plt.show()
        # plt.imshow(warped_image); plt.show()

        return True, warped_image


import torch
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from copy import deepcopy


class FeatureExtractor():
    def __init__(self):
        self.net = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(0).eval()
        self.feature_extractor = create_feature_extractor(self.net, {
            "layer1.0.bn1": "conv1",
            "layer1.0.bn2": "conv2",
            "layer1.1.bn1": "conv3",
            "layer1.1.bn2": "conv4",
        })
        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        x = torch.from_numpy(x.transpose(2,0,1)[None]).float().to(0)
        x = self.transforms(x)
        with torch.no_grad():
            features = list(self.feature_extractor(x).values())

        features = [F.resize(x, size=features[0].shape[2:]) for x in features]
        features = [x.squeeze().cpu().numpy().transpose(1,2,0) for x in features]
        features = np.concatenate(features, axis=2)
        return np.array(features)


class BackRemover():
    def __init__(self):
        self.warper = Warper()
        self.extractor = FeatureExtractor()

    def __call__(self, real, pred, pts_board, pts_center, mask_aux=None, upperlim=0.12, thresh=0.8):
        real_orig = deepcopy(real)
        pred_orig = deepcopy(pred)

        size = np.array(real.shape[-2:-4:-1])
        real = cv2.resize(real, size)
        pred = cv2.resize(pred, size)
        ret, out = self.warper(pred, real)

        if ret:
            pred = out
        else:
            msg = out
            return False, msg

        size = np.array(real.shape[-2:-4:-1]) // 4
        real = cv2.resize(real, size)
        pred = cv2.resize(pred, size)
        
        real = real.astype(np.float32) / 255.
        pred = pred.astype(np.float32) / 255.

        real = cv2.blur(real, (3,3))
        pred = cv2.blur(pred, (3,3))

        real = self.extractor(real)
        pred = self.extractor(pred)

        diff = np.mean(np.abs(real - pred), axis=2)

        # for tuning
        # print(diff.max(), np.percentile(diff, [98]))

        diff = np.clip(diff, 0, upperlim) / upperlim
        alpha = np.where(diff < thresh, 0, diff)
        alpha = (alpha.astype(np.float32) * 255).astype(np.uint8)
        alpha = cv2.resize(alpha, real_orig.shape[-2:-4:-1])

        # paint charuco board with black
        # alpha = cv2.drawContours(alpha, [pts_board], -1, 0, -1)

        # for debug
        diff = (diff * 255.).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

        # Open first!
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=10)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=10)
        alpha = cv2.erode(alpha, kernel, iterations=5)

        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for k, cnt in enumerate(contours):
            ret = cv2.pointPolygonTest(cnt, pts_center, False)
            if ret==1:
                mask = cv2.drawContours(mask, [cnt], 0, 255, -1)
        if mask_aux is not None:
            mask = mask * (mask_aux.astype(np.float32) / 255.)

        msg = check_protrude(mask)
        if msg is not None:
            return False, msg

        alpha = (alpha.astype(np.float32) / 255.) * (mask.astype(np.float32) / 255.)
        segm = (real_orig.astype(np.float32) / 255.) * alpha.astype(np.float32)[...,None]
        segm = (segm * 255.).astype(np.uint8)
        alpha = (alpha * 255.).astype(np.uint8)

        mask_aux = cv2.threshold(alpha, int(255*0.9), 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(144,144))
        mask_aux = cv2.dilate(mask_aux, kernel, iterations=1)

        return True, segm, mask_aux, diff, alpha

    
def check_protrude(mask):
    ret1 = np.any(mask[[0,-1]] == 255) or np.any(mask[:, [0,-1]] == 255)
    ret2 = np.all(mask == 0)
    if ret1 or ret2:
        return "check_protrude error!"
    else:
        return None


def project_w2c(pts, rvec, tvec, mtx, dist):
    pts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
    pts = pts.squeeze()
    return pts


def crop(img, cpos_target, size, scale):
    out = deepcopy(img)

    h, w = out.shape[:2]
    out = cv2.resize(out, (int(w * scale), int(h * scale)))

    cx, cy = (cpos_target * scale).squeeze().astype(np.int64).tolist()
    h, w = out.shape[:2]

    # padding for centering
    padx = w - cx * 2
    padx = (padx, 0) if padx > 0 else (0, -padx)
    pady = h - cy * 2
    pady = (pady, 0) if pady > 0 else (0, -pady)

    if len(img.shape) == 3:
        out = np.pad(out, [pady, padx, (0, 0)])
        out = np.pad(out, [(size, size), (size, size), (0, 0)])
    else:
        out = np.pad(out, [pady, padx])
        out = np.pad(out, [(size, size), (size, size)])

    _h, _w = out.shape[:2]
    x0 = int(_w / 2 - size / 2)
    y0 = int(_h / 2 - size / 2)
    out = out[y0:y0+size, x0:x0+size]
    return out