import torch
import logging

import models.modules.GSANet as GSANet

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    if which_model == 'GSANet':
        netG = GSANet.Model_G(32)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG