import os
import cv2
import numpy as np


def IoU(prediction, label, smooth=0.0000001):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    out = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return out


label_dir = r'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\zhangyongsurf-20250213-upload\upload\upload_prediction\SDSaliency900\label'
prediction_dir = r"F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\zhangyongsurf-20250213-upload\upload\upload_prediction\SDSaliency900"



methods_list = os.listdir(prediction_dir)
for method in methods_list:
    method_dir = os.path.join(prediction_dir, method)
    files = os.listdir(label_dir)
    IoU_list = []
    for f in files:
        label_path = os.path.join(label_dir, f)
        label = cv2.imread(label_path)
        pre_path = os.path.join(method_dir, f)


        pre = cv2.imread(pre_path)

        mask = label[:, :, 0]
        pre = pre[:, :, 0]
        pre = pre / 255
        mask[mask > 0] = 1
        pre[pre >= 0.5] = 1
        pre[pre < 0.5] = 0

        IoU_score = IoU(prediction=pre, label=mask)

        IoU_list.append(IoU_score)
    IoU_list = np.array(IoU_list)
    print(method, np.mean(IoU_list))
