from utils import *
import numpy as np
import cv2
import cv2.aruco as aruco
import torch
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import imageio
from copy import deepcopy


class Warper():
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher()

    def __call__(self, float_img, ref_img):
        float_img = deepcopy(float_img)
        ref_img = deepcopy(ref_img)
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
    def __init__(self, thresh=0.1):
        self.extractor = FeatureExtractor()
        self.thresh = thresh

    def __call__(self, real, pred, cpos_target):
        orig = deepcopy(real)
        real = deepcopy(real)
        pred = deepcopy(pred)

        size = np.array(real.shape[-2:-4:-1]) // 4
        real = cv2.resize(real, size)
        pred = cv2.resize(pred, size)
        real = cv2.blur(real, (3,3))
        pred = cv2.blur(pred, (3,3))

        real = real.astype(np.float32) / 255.
        pred = pred.astype(np.float32) / 255.
        real = self.extractor(real)
        pred = self.extractor(pred)

        diff = np.mean(np.abs(real - pred), axis=2)

        # for tuning
        # plt.figure(figsize=(4,2)); plt.hist(diff.flatten(), bins=100); plt.show()

        alpha = np.where(diff < self.thresh, 0, 1)
        # alpha = np.where(diff < self.thresh, 0, diff)
        # alpha = np.clip(alpha, 0, self.upperlim) / self.upperlim
        alpha = (alpha.astype(np.float32) * 255).astype(np.uint8)
        alpha = cv2.resize(alpha, orig.shape[-2:-4:-1])

        # paint charuco board with black
        # wpos_board = np.float32([[3.5,-2,0], [3.5,-7,0], [-3.5,-7,0], [-3.5,-2,0]]) * sl
        # cpos_board = project_w2c(wpos_board, rvec, tvec, mtx, dist).astype(np.int32)
        # alpha = cv2.drawContours(alpha, [pts_board], -1, 0, -1)

        # for debug
        diff = ((diff - diff.min()) / (diff.max() - diff.min()) * 255.).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=10)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=10)
        alpha = cv2.erode(alpha, kernel, iterations=5)

        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        ret = -1.
        for k, cnt in enumerate(contours):
            ret = cv2.pointPolygonTest(cnt, cpos_target, False)
            if ret==1:
                mask = cv2.drawContours(mask, [cnt], 0, 255, -1)
        if ret == -1.:
            return False, "no target error!"
        # if mask_aux is not None:
        #     mask = mask * (mask_aux.astype(np.float32) / 255.)

        alpha = (alpha.astype(np.float32) / 255.) * (mask.astype(np.float32) / 255.)
        segm = (orig.astype(np.float32) / 255.) * alpha.astype(np.float32)[...,None]
        segm = (segm * 255.).astype(np.uint8)
        alpha = (alpha * 255.).astype(np.uint8)

        mask_aux = cv2.threshold(alpha, int(255*0.9), 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(144,144))
        # mask_aux = cv2.dilate(mask_aux, kernel, iterations=1)

        return True, segm, mask_aux, diff, alpha


class Cropper():
    def __init__(self, size=512, scale=1.):
        self.size = 512
        self.scale = scale

    def __call__(self, img, cpos_target, wpos_camera):
        scale = np.sum(wpos_camera**2)**0.5 * self.scale

        cx, cy = cpos_target
        h, w = img.shape[:2]

        _size = int(self.size / scale)
        x0 = int(cx - _size / 2)
        y0 = int(cy - _size / 2)
        x1 = x0 + _size
        y1 = y0 + _size

        if x0 < 0 or y0 < 0 or x1 >= w or y1 >=h:
            return False, "crop error!"

        img = img[y0:y0+_size, x0:x0+_size]
        img = cv2.resize(img, (self.size, self.size))
        return True, img, (x0, y0, x1, y1)


import cv2
import cv2.aruco as aruco
import imageio
import numpy as np
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy
import pickle
import os

if __name__ == '__main__':
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm


class Preprocessor():
    def __init__(self, mtx, dist, save_dir='data', thresh=0.1, size=512, scale=1., debug=False):
        self.mtx = mtx
        self.dist = dist
        self.size = size

        self.warper = Warper()  # very important!
        self.remover = BackRemover(thresh)
        self.cropper = Cropper(size, scale)

        self.frames = []
        self.views = []

        self.debug = debug
        self.T = 11
        self.N = 0

        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'video'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'video_large'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'views'), exist_ok=True)

    def __call__(self, indices, main_real, main_pred, rvec_hist, tvec_hist):
        bad_indices = []
        index_prev = -1  # last success index
            
        for t, (index, real, pred, rvec, tvec) in enumerate(
            zip(tqdm(indices), main_real, main_pred, rvec_hist, tvec_hist)):

            real_orig, real = deepcopy(real), deepcopy(real)
            pred_orig, pred = deepcopy(pred), deepcopy(pred)

            if not real.shape == pred.shape:
                pred = cv2.resize(pred, real.shape[-2:-4:-1])

            R, T = cv2.Rodrigues(rvec)[0], tvec

            wpos_target = np.float32([[0,0,0.065]])
            cpos_target = project_w2c(wpos_target, rvec, tvec, self.mtx, self.dist).astype(np.int16)

            wpos_camera = np.dot(R.T, - T).squeeze() - wpos_target.squeeze()
            wrot_camera = cv2.RQDecomp3x3(R.T)[0]

            ret, *out = self.warper(pred, real)
            if ret:
                pred, = out

                ret, *out = self.remover(real, pred, cpos_target)    
                if ret:
                    segm, mask_aux, diff, alpha = out

                    ret, *out = self.cropper(segm, cpos_target, wpos_camera)
                    if ret:
                        crop, (x0, y0, x1, y1) = out

            if not ret:
                msg, = out
                bad_indices.append(index)
                print("skip (t={}, msg=\"{}\")".format(t, msg))

            else:
                view = wpos_camera / np.linalg.norm(wpos_camera)
                self.frames.append(crop)
                self.views.append(view)

                if len(self.frames) == self.T:
                    assert len(self.frames) == len(self.views)
                    frames_4 = [cv2.resize(frame, (self.size//4,self.size//4)) for frame in self.frames]

                    path = os.path.join(self.save_dir, "video", "{:04}.gif".format(self.N))
                    imageio.mimwrite(path, self.frames)
                    path = os.path.join(self.save_dir, "video_large", "{:04}.gif".format(self.N))
                    imageio.mimwrite(path, frames_4)
                    path = os.path.join(self.save_dir, "views", "{:04}.npy".format(self.N))
                    np.save(path, np.array(self.views))
                    
                    self.frames = []
                    self.views = []
                    self.N += 1

                index_prev = index

            if self.debug:
                if ret and t % 5 == 0:
                    segm2 = deepcopy(segm)
                    segm2 = cv2.rectangle(segm2,(x0, y0),(x1, y1),color=255,thickness=5)
                    plt.figure(figsize=(20,20))
                    plt.subplot(1,5,1); plt.imshow(diff)
                    plt.subplot(1,5,2); plt.imshow(alpha)
                    plt.subplot(1,5,3); plt.imshow(segm)
                    plt.subplot(1,5,4); plt.imshow(segm2)
                    plt.subplot(1,5,5); plt.imshow(crop)
                    plt.show()

                # first failure in last some frames
                if not ret and index_prev == index - 1:
                    plt.figure(figsize=(8,8))
                    plt.subplot(1,2,1); plt.imshow(real_orig)
                    plt.subplot(1,2,2); plt.imshow(pred_orig)
                    plt.show()

        print("bad_indices:", bad_indices)


if __name__ == '__main__':
    with open('camera_undist.pkl', 'rb') as f:
        (mtx, dist) = pickle.load(f)

    preprocessor = Preprocessor(mtx, dist)

    for idx in range(15):
        print(idx)

        main_real = imageio.mimread("temp/main_real_{:03}.mp4".format(idx), memtest=False)
        main_pred = imageio.mimread("temp/main_pred_{:03}.mp4".format(idx), memtest=False)

        with open('temp/poses_{:03}.pkl'.format(idx), 'rb') as f:
            (indices, rvec_hist, tvec_hist) = pickle.load(f)

        assert len(main_real) == len(main_pred) == len(rvec_hist) == len(tvec_hist)

        preprocessor(indices, main_real, main_pred, rvec_hist, tvec_hist)

        print('Number of saved data:', preprocessor.N)