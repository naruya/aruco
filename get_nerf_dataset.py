# python get_nerf_dataset.py --vid vid.mp4

import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import pickle
import imageio
import PIL
import json
import torch
from utils import undistort, estimate_pose


parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, default="./")
parser.add_argument("--skip", type=int, default=6)
parser.add_argument("--drop", type=int, default=30)
parser.add_argument("--squares", type=int, nargs="+", default=[5,7])
parser.add_argument("--mode", type=str, default='A4')
parser.add_argument("--sl", type=int, default=0.04)
parser.add_argument("--ml", type=int, default=0.02)
args = parser.parse_args()

print(args)

vid = imageio.mimread(args.vid, memtest=False)
vid = vid[args.drop:-args.drop:args.skip]
print('vid: ', len(vid), vid[0].shape)

with open('camera_raw.pkl', 'rb') as f:
    (mtx, dist) = pickle.load(f)

vid = undistort(vid, mtx, dist)

with open('camera_undist.pkl', 'rb') as f:
    (mtx, dist) = pickle.load(f)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
charuco = aruco.CharucoBoard_create(
    squaresX=args.squares[0],
    squaresY=args.squares[1],
    squareLength=args.sl,  # no need to change
    markerLength=args.ml,  # no need to change
    dictionary=aruco_dict)


# get pose
indices, rvec_hist, tvec_hist = estimate_pose(
    vid, aruco_dict, charuco, mtx, dist, args.sl, args.mode)

vid_sample = list(np.array(vid)[indices])
print(len(vid_sample), len(rvec_hist), len(tvec_hist))

if 'main' in os.path.basename(args.vid):
    path = os.path.basename(args.vid).replace('main', 'main_real')
    imageio.mimwrite(path, vid_sample, macro_block_size=8)
    path = os.path.basename(args.vid).replace('main', 'poses').replace('mp4', 'pkl')
    with open(path, 'wb') as f:
        pickle.dump((indices, rvec_hist, tvec_hist), f)

H, W = vid[0].shape[:2]

if 'back' in os.path.basename(args.vid):
    os.makedirs('charuco/images/', exist_ok=True)

out = {
    "fl_x": float(mtx[0,0]),
    "fl_y": float(mtx[1,1]),
    "cx": float(mtx[0,2]),
    "cy": float(mtx[1,2]),
    "w": W,
    "h": H,
    "camera_model": 'OPENCV',
    "k1": dist[0,0],
    "k2": dist[0,1],
    "p1": dist[0,2],
    "p2": dist[0,3],
}

poses = []
for t, (rvec, tvec) in enumerate(zip(rvec_hist, tvec_hist)):
    R, _ = cv2.Rodrigues(rvec)
    T = tvec

    c2w = np.eye(4)
    c2w[:3,3] = np.dot(R.T, - T).squeeze()
    c2w[:3,:3] = np.dot(R.T, np.array([[1,0,0], [0,-1,0], [0,0,-1]]))
    poses.append(c2w)

poses = torch.from_numpy(np.array(poses).astype(np.float32))
print('pose max:', torch.max(torch.abs(poses[:, :3, 3])))

frames = []
for t, (img, c2w) in enumerate(zip(vid_sample, poses)):
    name = 'images/frame_{:05}.png'.format(t)

    if 'back' in os.path.basename(args.vid):
        PIL.Image.fromarray(img).save('charuco/' + name, quality=95)  # heavy

    frame = {
        "file_path": name,
        "transform_matrix": c2w.tolist(),
    }
    frames.append(frame)

out["frames"] = frames

if 'main' in os.path.basename(args.vid):
    path = os.path.basename(args.vid).replace('main', 'transforms').replace('mp4', 'json')
else:
    path = "charuco/transforms.json"

with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4)