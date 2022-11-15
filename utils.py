import cv2
import cv2.aruco as aruco
import numpy as np
from copy import deepcopy
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

        if response < 6:  # (expected: 'count >= 6')
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
    return [_undistort(img) for img in vid]


def estimate_poses(vid, aruco_dict, charuco, mtx, dist, sl, mode='A4'):
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
                objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[-2,3.5,0]]) * sl,
                               np.array([[0,-1,0], [-1,0,0], [0,0,-1]]))
            elif mode=='A3':
                objpi = np.dot(charuco.chessboardCorners[idx] - np.array([[5,7,0]]) * sl,
                               np.array([[-1,0,0], [0,1,0], [0,0,-1]]))
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


def project_w2c(pts, rvec, tvec, mtx, dist):
    pts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
    return pts.squeeze()