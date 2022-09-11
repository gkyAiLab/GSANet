import os
import shutil


path = './model_20220324_1400000_G'
save_path = './process_final_result'
if not os.path.exists(save_path):
    os.mkdir(save_path)


for i in range(201):
    if (i + 3) % 3 == 0:
        shutil.copy(os.path.join(path, "{:04d}.png".format(i)), os.path.join(save_path, "{:04d}.png".format(i)))
        shutil.copy(os.path.join(path, "{:04d}_alignexposures.npy".format(i)), os.path.join(save_path, "{:04d}_alignexposures.npy".format(i)))
        