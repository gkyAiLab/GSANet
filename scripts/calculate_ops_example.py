import torch
import torch.nn as nn
import numpy as np
from complexity_metrics import get_gmacs_and_params, get_runtime
from toy_model import ToyHDRModel
import argparse
from data_io import imread_uint16_png
import cv2
import sys
sys.path.append("..")

import codes.models.modules.GSANet as GSANet



parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--write_path", type=str, default="./", help="Path to write the readme.txt file")
parser.add_argument("-rp", "--read_path", type=str, default="./", help="Path to the training dataset")
args = parser.parse_args()
write_path=args.write_path
read_path = args.read_path

"""
Note: The inputs to the model should only be the provided unprocessed input data.
All processing of inputs (e.g. gamma correction, normalization) should be done within the model definition.
In this example the three input images are concatenated and passed in as the first tensor, and the exposure values are passed in as the second tensor.
This has an input shape of [(1, 3, 3, 1060, 1900), (1, 3)] corresponding to [(batch_size, num images, channels, height, width), (batch_size, exposure_values)] 
"""

image_id = 0
# Read input triplets
image_short = cv2.cvtColor(cv2.imread(read_path + "{:04d}_short.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
image_medium = cv2.cvtColor(cv2.imread(read_path + "{:04d}_medium.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
image_long = cv2.cvtColor(cv2.imread(read_path + "{:04d}_long.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

# Convert to channels first pytorch tensor
input_images = torch.FloatTensor(np.array([image_short, image_medium, image_long])).permute(0,3,1,2).unsqueeze(0)

# Read exposures and find relation to the medium frame (i.e. GT aligned to medium frame)
exposures=np.load(read_path + "{:04d}_exposures.npy".format(image_id))
floating_exposures = exposures - exposures[1]
input_exposures = torch.FloatTensor(floating_exposures).unsqueeze(0)

# Load a pytorch model
# model = ToyHDRModel()
model = GSANet.Model_G(32)
model.eval()

# Calculate MACs and Parameters
total_macs, total_params = get_gmacs_and_params(model, input_size=[(1, 3, 3, 1060, 1900), (1, 3)])

# Calculate Runtime 
# mean_runtime = get_runtime(model, input_tensors=[input_images, input_exposures])

print(total_macs)
print(total_params)
print(mean_runtime)

# Print model statistics to txt file
with open(write_path + 'readme.txt', 'w') as f:
    f.write("runtime per image [s] : " + str(mean_runtime))
    f.write('\n')
    f.write("number of operations [GMAcc] : " + str(total_macs))
    f.write('\n')
    f.write("number of parameters  : " + str(total_params))
    f.write('\n')
    f.write("Other description: Toy Model for demonstrating example code usage.")
# Expected output of the readme.txt for ToyHDRModel should be:
# runtime per image [s] : 0.013018618555068967
# number of operations [GMAcc] : 20.146042
# number of parameters  : 8243
# Other description: Toy Model for demonstrating example code usage.

print("You reached the end of the calculate_ops_example.py demo script. Good luck participating in the NTIRE 2022 HDR Challenge!")

