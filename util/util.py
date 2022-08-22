
import numpy as np
import cv2
import os

def read_image(path, norm_val = None):

    if norm_val == (2**16-1):
        frame = cv2.imread(path, -1)
        frame = frame / norm_val
        frame = frame[...,::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame = frame / 255.
    return np.expand_dims(frame, axis = 0)


def crop_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]


def make_lf_aif_gt_dataset(img_list,dir):
    aif_gt_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for f in img_list:
        aif_file = os.path.split(f)[-1].split('_ap')[0]
        aif_file_name_tmp = aif_file + '.png'
        aif_file_name = os.path.join(dir, aif_file_name_tmp)
        if os.path.exists(aif_file_name):
            aif_gt_files.append(aif_file_name)
    return aif_gt_files
