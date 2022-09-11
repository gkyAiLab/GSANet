
import cv2
import numpy as np
import data_io as io

def tonemapped_results(img_path, alignratio_path, save_path):
    img = io.imread_uint16_png(img_path, alignratio_path)
    img = np.clip(np.tanh(img), 0, 1) * 255
    img = img.round().astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    img_path = r'D:\A-Document\Code\Github\GSANet\results\GSANet_0822\GSANet_latest_G\0170.png'
    alignratio_path = r'D:\A-Document\Code\Github\GSANet\results\GSANet_0822\GSANet_latest_G\0170_alignexposures.npy'
    save_path = r'D:\A-Document\Code\Github\GSANet\results\GSANet_0822\GSANet_latest_G\results\0170_tonemapped.png'
    
    tonemapped_results(img_path, alignratio_path, save_path)

    print('finished!!')
