import caffe
import numpy as np
import os
from PIL import Image
import sys
sys.path.append("..")
from pyutils import score_refine_net_v2 as score

"""
for VOC
sys.argv[1]: iter (60000)
sys.argv[2]: voc dataset root
sys.argv[3]: the result path
sys.argv[4]: gpu id, default 0
"""
test_proto = '../models/voc_rnn_deploy_vgg_v3.prototxt'
test_model = '../models/refine/vgg_scratch_c7_iter_{}.caffemodel'.format(sys.argv[1])
layer = 'prob'
dataroot = '{}/VOC/pascal/VOCdevkit/VOC2012'.format(sys.argv[2])
# prior_root = '/media/sifeil/NV_share/Results/davis_global_JC3/ResNetF_perobj_27000/'
result_folder = '{}'.format(sys.argv[3])
if len(sys.argv) > 4:
	caffe.set_device[sys.argv[4]]
else:
	caffe.set_device(0)
caffe.set_mode_gpu()
cnnnet = caffe.Net(test_proto, test_model, caffe.TEST)
val = np.loadtxt('{}/ImageSets/Segmentation/val.txt'.format(dataroot), dtype=str)
score.seg_tests(rnnnet, sys.argv[1], dataroot, result_folder, val, False, layer='prob')