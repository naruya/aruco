# python get_intrinsics.py --vid charuco.mp4

import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import pickle
import imageio
import matplotlib.pyplot as plt
from utils import calibrate, undistort


parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, default="./")
parser.add_argument("--out", type=str, default="./")
parser.add_argument("--skip", type=int, default=6)
parser.add_argument("--drop", type=int, default=30)
parser.add_argument("--squares", type=int, nargs="+", default=[5,7])
args = parser.parse_args()

print(args)

os.makedirs(args.out, exist_ok=True)

vid = imageio.mimread(args.vid, memtest=False)

print('vid: ', len(vid), vid[0].shape)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
charuco = aruco.CharucoBoard_create(
    squaresX=args.squares[0],
    squaresY=args.squares[1],
    squareLength=0.04,  # no need to change
    markerLength=0.02,  # no need to change
    dictionary=aruco_dict)

print('camera_raw')
cameraMatrix, distCoeffs = calibrate(
    vid[args.drop:-args.drop:args.skip], aruco_dict, charuco)
with open(os.path.join(args.out, 'camera_raw.pkl'), 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs), f)

vid_undist = undistort(
    vid[args.drop:-args.drop:args.skip], cameraMatrix, distCoeffs)

print('camera_undist')
cameraMatrix, distCoeffs = calibrate(vid_undist, aruco_dict, charuco)
with open(os.path.join(args.out, 'camera_undist.pkl'), 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs), f)

# for debug
imageio.mimwrite("temp_undist{:04}.mp4".format(np.random.randint(1000)), vid_undist, macro_block_size=8)
