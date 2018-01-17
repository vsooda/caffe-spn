from __future__ import division
import caffe
import numpy as np
import os
import sys
import scipy.io
import time
from datetime import datetime
from PIL import Image
import cv2

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

# All of the following functions are maintained with uint8 0-255

def image_padding(im_array, pad_dim):
    if im_array.shape[0] > pad_dim[0] or im_array.shape[1] > pad_dim[1]:
        im_pad = cv2.resize(im_array, (pad_dim[1], pad_dim[0]))
        bound_x=0
        bound_y=0
    else:
        im_pad = np.zeros((pad_dim[0],pad_dim[1],im_array.shape[2]))
        bound_y = int((pad_dim[0]-im_array.shape[0])/2)
        bound_x = int((pad_dim[1]-im_array.shape[1])/2)
        im_pad[bound_y:bound_y+im_array.shape[0], bound_x:bound_x+im_array.shape[1],:] = im_array
    return im_pad, (bound_y,bound_x)


def prior_padding(im_array, pad_dim):
    if im_array.shape[0] > pad_dim[0] or im_array.shape[1] > pad_dim[1]:
        im_pad = cv2.resize(im_array, (pad_dim[1], pad_dim[0]))
        bound_x=0
        bound_y=0
    else:
        im_pad = np.zeros((pad_dim[0],pad_dim[1],im_array.shape[2]))
        im_pad[:,:,0] = 1
        bound_y = int((pad_dim[0]-im_array.shape[0])/2)
        bound_x = int((pad_dim[1]-im_array.shape[1])/2)
        im_pad[bound_y:bound_y+im_array.shape[0], bound_x:bound_x+im_array.shape[1],:] = im_array
    return im_pad, (bound_y,bound_x)


def im_from_padding(im_pad, im_dim, bound):
    if im_dim[0] > im_pad.shape[0] or im_dim[1] > im_pad.shape[1]:
        im_crop = cv2.resize(im_pad, (im_dim[1],im_dim[0]))
    else:
        im_crop = im_pad[bound[0]:bound[0]+im_dim[0], bound[1]:bound[1]+im_dim[1],:]
    return im_crop


def compute_hist(net, dataroot, save_dir, dataset, layer='score'):
    n_cl = net.blobs[layer].channels
    if save_dir and not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    # loss = 0
    for idx in dataset:
        print idx
        im = Image.open('{}/pascal/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(dataroot,idx))
        lb = Image.open('{}/pascal/VOCdevkit/VOC2012/SegmentationClass/{}.png'.format(dataroot,idx))
        im_dim = (im.height,im.width)
        lb = np.array(lb,dtype=np.uint8)
        lb = lb[:,:,np.newaxis]


        rf = scipy.io.loadmat('{}/pascal/VOCdevkit/VOC2012/deeplabv2-resnet101/{}.mat'.format(dataroot,idx))

        rf = np.array(rf['out'], dtype=np.float32)
        hard_rf = rf.argmax(0)
        valid_lbs = np.unique(hard_rf[hard_rf<22])
        rf = rf.transpose((1,2,0))
        prior = prior_padding(rf, (512,512))[0]

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))

        in_, bound = image_padding(in_, (512,512))

        trans_in_ = in_.transpose((2,0,1))
        trans_lb_ = lb.transpose((2,0,1))
        trans_rf_ = prior.transpose((2,0,1))
        image = np.append(trans_in_,trans_rf_,0)

        net.blobs['data'].reshape(1, *image.shape)
        net.blobs['data'].data[...] = image
        net.forward()
        prob = net.blobs[layer].data[0]
        prob_ = prob.transpose((1,2,0))
        prob_ = im_from_padding(prob_, im_dim, bound)
        prob_ = prob_.transpose((2,0,1))

        prob_rect = np.zeros_like(prob_)
        prob_rect[valid_lbs] = prob_[valid_lbs]
        
        prob_hard = prob_rect.argmax(0)
        hist += fast_hist(trans_lb_.flatten(),
                                prob_hard.flatten(),
                                n_cl)


        im = Image.fromarray(prob_hard.astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, idx + '.png'))
        scipy.io.savemat(os.path.join(save_dir, idx + '.mat'), dict(prob = prob_rect))
    return hist



def seg_tests(net, iter, dataroot, save_format, dataset, layer='score'):
    print '>>>', datetime.now(), 'Begin seg tests'
    n_cl = net.blobs[layer].channels
    hist = compute_hist(net, dataroot, save_format, dataset, layer)

    # mean loss
    # print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
