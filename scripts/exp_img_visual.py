import cv2
import numpy as np
import os.path as osp

def read_img(path, ratio=255.0):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)

def expo_correct(img, exposures, idx):
    floating_exposures = exposures - exposures[1]
    gamma=2.24
    img_corrected = (((img**gamma)*2.0**(-1*floating_exposures[idx]))**(1/gamma))
    return img_corrected


def exp_img_visual(img_path, exp_path, idx):
    ## get image
    img = read_img(img_path)

    ## get exposures
    exposures = np.load(exp_path)

    ## save path
    save_path = '/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/lfy/HDR/HDR_dataset/exp_img_visual/'

    if idx == 0:
        exp_img = expo_correct(img, exposures, idx=0)
        save_name = osp.basename(img_path)[:4] + '_exp_short_img.png'
    else:
        exp_img = expo_correct(img, exposures, idx=2)
        save_name = osp.basename(img_path)[:4] + '_exp_long_img.png'

    save_path = save_path + save_name
    # print(exp_img[0][0])
    uint8_image = np.round(exp_img* 255).astype(np.uint8)

    cv2.imwrite(save_path, uint8_image)

if __name__ == '__main__':
    img_path = '/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/lfy/HDR/HDR_dataset/exp_img_visual/0211_long.png'
    # img_path = '/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/lfy/HDR/HDR_dataset/train_1494/short/0200_short.png'
    exp_path = '/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/lfy/HDR/HDR_dataset/exp_img_visual/0211_exposures.npy'
    # idx = 0
    idx = 2
    exp_img_visual(img_path, exp_path, idx)

    