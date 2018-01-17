import caffe
import numpy as np
import math
from PIL import Image
import cv2
import scipy.io
import random
import sys

def Affinemat(angle, sx, sy, x0, y0, center = None, new_center = None):
    #angle = -angle/180.0*math.pi
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    if center is None:
        x = 0
        y = 0
    else:
        x = center[0]
        y = center[1]
    if new_center is None:
        nx = 0
        ny = 0
    else:
        nx = new_center[0]
        ny = new_center[1]
    a = cosine / sx
    b = sine / sx
    c = x-nx*a-ny*b
    d = -sine / sy
    e = cosine /sy
    f = y-nx*d-ny*e
    return (a,b,c,d,e,f)

class AgRcProbVOCDataLayer(caffe.Layer):
    """
    Load (input image, cnn output, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a rnn refine network with multiple label at once.

    we crop #patch_num# patches out of #sample_size# images in each mini batch
    """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.shape = np.array(params['shape'])
        self.patch_num = np.array(params['patch_num'])
        self.sample_size = self.shape[0] / self.patch_num
        self.crop_dims = (self.shape[2], self.shape[3])

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = np.array(range(self.sample_size))

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False
            for id in range(len(self.idx)):
                self.idx[id] = id

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            for id in range(len(self.idx)):
                self.idx[id] = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        self.data = np.zeros(self.shape)
        self.label = np.zeros((self.shape[0],1,self.shape[2],self.shape[3]))

        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)


    def forward(self, bottom, top):
        for id in range(len(self.idx)):
            # load image + label image pair
            image, label= \
            self.load_imagelabel_ac(self.indices[self.idx[id]])
            self.data[id*10:(id+1)*10], self.label[id*10:(id+1)*10] = \
                self.crop_random(image, label)

        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random:
            for id in range(len(self.idx)):
                self.idx[id] = random.randint(0, len(self.indices)-1)
        else:
            self.idx += self.shape[0]
            if self.idx[-1] >= len(self.indices):
                self.idx = range(self.shape[0])


    def backward(self, top, propagate_down, bottom):
        pass


    def load_imagelabel_ac(self, idx):
        shape = self.shape
        width_ = np.array(1.2*shape[3],dtype=int)
        height_ = np.array(1.2*shape[2],dtype=int)
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        lb = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))

        rf = scipy.io.loadmat('{}/SegmentationProb/{}.mat'.format(self.voc_dir, idx))
        rf = np.array(rf['out'], dtype=np.float32)
        rf = rf.transpose((1,2,0))

        (x,y) = im.size
        center = (x/2,y/2)
        rate = (np.random.rand(1)-0.5)
        shift_x = np.maximum(x,y) * rate
        rate = (np.random.rand(1)-0.5)
        shift_y = np.maximum(x,y) * rate
        scale_x = 1+(np.random.rand(1)-0.5) /5.0
        scale_y = 1+(np.random.rand(1)-0.5) /5.0
        angle = (np.random.rand(1)-0.5)*(30.0/180.0)*math.pi
        mat = Affinemat(angle,scale_x,scale_y,center,(center[0]+shift_x,center[1]+shift_y))

        im = im.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
        lb = lb.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
        if x < width_ or y < height_:
            im = im.resize((width_,height_),resample=Image.BILINEAR)
            lb = lb.resize((width_,height_),resample=Image.BILINEAR)
        image = np.array(im,dtype=np.float32)
        image = image[:,:,::-1]
        image -= self.mean
        label = np.array(lb,dtype=np.uint8)
        label = label[:,:,np.newaxis]


        for ch in range(rf.shape[2]):
            if ch==0:
                rf_tmp = 1-rf[:,:,ch]
            else:
                rf_tmp = rf[:,:,ch]
            rf_tmp = np.array(rf_tmp * 255.0,dtype=np.uint8)
            rf_tmp = Image.fromarray(rf_tmp)
            rf_tmp = rf_tmp.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
            if x < width_ or y < height_:
                rf_tmp = rf_tmp.resize((width_,height_),resample=Image.BILINEAR)
            rf_tmp = np.array(rf_tmp,dtype=np.float32) / 255.0
            if ch==0:
                rf_tmp = 1-rf_tmp
                prior = np.zeros((rf_tmp.shape[0],rf_tmp.shape[1],rf.shape[2]))
            prior[:,:,ch] = rf_tmp
        image = np.append(image,prior,2)

        return image, label

    def crop_random(self, image, label):  
        im_shape = np.array(image.shape)
        crop_dims = np.array(self.crop_dims,dtype=int)
        images = np.zeros((crop_dims[0],crop_dims[1],im_shape[2],10))
        labels = np.zeros((crop_dims[0],crop_dims[1],1,10))
        count = 0
        protect = 0
        if len(np.unique(label[label<21])) < 2:
            mmp = 0
        else:
            mmp = 1

        while count<self.patch_num:
            top = np.random.rand(1)* (im_shape[0]-crop_dims[0])
            left = np.random.rand(1)* (im_shape[1]-crop_dims[1])
            top = int(top)
            left = int(left)
            tmp_lb = label[top:top+crop_dims[0],left:left+crop_dims[1]]
            nidx = np.unique(tmp_lb[tmp_lb<21])
            if (len(nidx) < 2 and mmp) and protect < 10:
                protect += 1
                continue
            protect = 0
            images[:,:,:,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],:]
            labels[:,:,:,count] = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
            count += 1

        images = np.array(images, dtype=np.float32)
        images = images.transpose((3,2,0,1))
        labels = np.array(labels, dtype=np.uint8)
        labels = labels.transpose((3,2,0,1))
        return images, labels