import torch
import mindspore

dict_list = []
# for s1, s2 in zip(torch.load('rexnetv1_100-1b4dddf4.pth').items(), mindspore.load_checkpoint('rexnet.ckpt').items()):
    # if s1[0] != s2[0]:
    #     print(s1[0], s2[0])
    # exit()
    # print(s1[0])
for name, value in torch.load('rexnetv1_100-1b4dddf4.pth').items():
    param_dict = {}
    if name.endswith('num_batches_tracked'):
        continue
    elif name.endswith('weight'):
        if name.endswith('conv.weight'):
            name = name.replace('conv.weight', 'features.0.weight')
        elif name.endswith('bn.weight'):
            if name.endswith('se.bn.weight'):
                name = name.replace('weight', 'gamma')
            else:
                name = name.replace('bn.weight', 'features.1.gamma')
        elif name.endswith('fc1.weight'):
            name = name.replace('fc1.weight', 'conv_reduce.weight')
        elif name.endswith('fc2.weight'):
            name = name.replace('fc2.weight', 'conv_expand.weight')
        elif name.endswith('fc.weight'):
            name = name.replace('head.fc.weight', 'cls.1.weight')
    elif name.endswith('bias'):
        if name.endswith('bn.bias'):
            if name.endswith('se.bn.bias'):
                name = name.replace('bias', 'beta')
            else:
                name = name.replace('bn.bias', 'features.1.beta')
        elif name.endswith('fc1.bias'):
            name = name.replace('fc1.bias', 'conv_reduce.bias')
        elif name.endswith('fc2.bias'):
            name = name.replace('fc2.bias', 'conv_expand.bias')
        elif name.endswith('fc.bias'):
            name = name.replace('head.fc.bias', 'cls.1.bias')
    elif name.endswith('running_mean'):
        if name.endswith('se.bn.running_mean'):
            name = name.replace('running_mean', 'moving_mean')
        else:
            name = name.replace('bn.running_mean', 'features.1.moving_mean')
    elif name.endswith('running_var'):
        if name.endswith('se.bn.running_var'):
            name = name.replace('running_var', 'moving_variance')
        else:
            name = name.replace('bn.running_var', 'features.1.moving_variance')
    param_dict['name'] = name
    param_dict['data'] = mindspore.Tensor(value.numpy(), mindspore.float32)
    dict_list.append(param_dict)
    
mindspore.save_checkpoint(dict_list, 'rexnet.ckpt')
# for name, value in mindspore.load_checkpoint('rexnet.ckpt').items():
#     print(name)
# print(len(torch.load('rexnetv1_100-1b4dddf4.pth').items()), len(mindspore.load_checkpoint('rexnet.ckpt').items()))
