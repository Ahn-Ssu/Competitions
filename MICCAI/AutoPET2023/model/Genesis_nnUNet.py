import torch.nn as nn

################### configuration for model
num_input_channels=1
base_num_features = 32
num_classes = 2
net_num_pool_op_kernel_sizes=[[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[1, 2, 2]]
net_conv_kernel_sizes = [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]]
net_numpool=len(net_num_pool_op_kernel_sizes)
conv_per_stage = 2
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
##############################

model = Generic_UNet(num_input_channels, base_num_features, num_classes,
                                    net_numpool,
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)