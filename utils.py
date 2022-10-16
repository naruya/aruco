import cv2
import cv2.aruco as aruco
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def calibrate(vid, aruco_dict, charuco):
    corners_all = []
    ids_all = []

    for t, img in enumerate(vid):
        img = deepcopy(img)
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=aruco_dict)

        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco)

        if response < 10:
            print("Not able to detect a charuco board in image: t={}".format(t))
            continue

        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

    print("Number of detected frames:", len(corners_all))

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=charuco,
            imageSize=vid[0].shape[1::-1],
            cameraMatrix=None,
            distCoeffs=None)

    print('ret\n', ret, '\nmtx\n', cameraMatrix, '\ndist\n', distCoeffs)

    return cameraMatrix, distCoeffs


def undistort(vid, mtx, dist):
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, vid[0].shape[1::-1], 0)
    _undistort = lambda img: cv2.undistort(deepcopy(img), mtx, dist, None, newmtx)
    return [_undistort(img) for img in tqdm(vid)]