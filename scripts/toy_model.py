import torch
import torch.nn as nn

# Toy HDR Model
class ToyHDRModel(nn.Module):

    def __init__(self):
        super(ToyHDRModel, self).__init__()
      
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16*3, 16, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
    
    def LDR2HDR(self, img, expo, gamma=2.2):
        #Map the LDR input image to the HDR domain
        return img ** gamma / expo

    def forward(self, X, exposure_values):

        # for a single test scene, the input tensor X has shape (1, 3, 3, 1060, 1900) - (batch_size, num images, channels, height, width) 
        x1 = X[:,0,:,:,:]
        x2 = X[:,1,:,:,:]
        x3 = X[:,2,:,:,:]

        # map the LDR inputs into the HDR domain
        x1_lin = self.LDR2HDR(x1, 2**exposure_values[:,0])
        x2_lin = self.LDR2HDR(x2, 2**exposure_values[:,1])
        x3_lin = self.LDR2HDR(x3, 2**exposure_values[:,2])

        # concatenate the LDR and HDR inputs in the channel dimension
        x1_ = torch.cat([x1, x1_lin], 1)
        x2_ = torch.cat([x2, x2_lin], 1)
        x3_ = torch.cat([x3, x3_lin], 1)

        # process the concatenated inputs with a simple cnn
        F1 = self.relu(self.conv1(x1_))
        F2 = self.relu(self.conv1(x2_))
        F3 = self.relu(self.conv1(x3_))
        F_cat = torch.cat((F1, F2, F3), 1)
        F_mid = self.conv2(F_cat)
        F_out = self.conv3(F_mid)
        HDR_out = self.relu(F_out)
        
        return HDR_out