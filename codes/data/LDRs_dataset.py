import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import random

class LDRs_dataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LDRs_dataset, self).__init__()
        self.opt = opt
        self.paths_LDRs = None
        self.LDRs_env = None  # environment for lmdb
        self.data_type = opt['data_type']
        # read image list from lmdb or image files
        print(opt['dataroot_LDRs'])
        self.sizes_ldr, self.paths_ldr = util.get_image_paths(self.data_type, opt['dataroot_LDRs'])
        # print(self.paths_ldr)

        self.paths_short_ldr = util.get_paths(opt['dataroot_LDRs'], '*_short.png')
        self.paths_medium_ldr = util.get_paths(opt['dataroot_LDRs'], '*_medium.png')
        self.paths_long_ldr = util.get_paths(opt['dataroot_LDRs'], '*_long.png')
        self.paths_exposures = util.get_paths(opt['dataroot_LDRs'], '*_exposures.npy')

        assert self.paths_short_ldr, 'Error: LDRs paths are empty.'

    def __getitem__(self, index):
        # short_ldr_path = None
        # get exposures
        exposures = np.load(self.paths_exposures[index])
        floating_exposures = exposures - exposures[1]

        # get LDRs image
        ldr_images = []
        short_ldr_paths = self.paths_short_ldr[index]
        short_ldr_images = util.read_imgdata(short_ldr_paths, ratio=255.0)

        medium_ldr_paths = self.paths_medium_ldr[index]
        medium_ldr_images = util.read_imgdata(medium_ldr_paths, ratio=255.0)

        long_ldr_paths = self.paths_long_ldr[index]
        long_ldr_images = util.read_imgdata(long_ldr_paths, ratio=255.0)

        H, W, C = short_ldr_images.shape

        ldr_images.append(short_ldr_images)
        ldr_images.append(medium_ldr_images)
        ldr_images.append(long_ldr_images)
        ldr_images = np.array(ldr_images)

        img0 = ldr_images[0].astype(np.float32).transpose(2, 0, 1)
        img1 = ldr_images[1].astype(np.float32).transpose(2, 0, 1)
        img2 = ldr_images[2].astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)

        img0 =img0.unsqueeze(0) # torch.Size([1, 6, 256, 256])
        img1 =img1.unsqueeze(0)
        img2 =img2.unsqueeze(0)
        img_ldrs = torch.cat((img0, img1, img2))
       
        sample = {'img_LDRs': img_ldrs, 'float_exp':floating_exposures, 'short_path': short_ldr_paths}
        return sample
        
    def __len__(self):
        return len(self.paths_exposures)
