import os
import sys
import glob
import shutil
import logging
import functools
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from scipy import ndimage, misc, signal, spatial


def re_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_file(path_s, path_t):
    shutil.copy(path_s, path_t)    

def init_log(output_dir):
    re_mkdir(output_dir)
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        filename=os.path.join(output_dir, 'log.log'),
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def get_files_in_folder(folder, file_ext=None):
    files = glob.glob(os.path.join(folder, "*" + file_ext))
    #print 'debug_files',files
    files_name = []
    for i in files:
        _, name = os.path.split(i)
        name, ext = os.path.splitext(name)
        files_name.append(name)
    #print np.asarray(files),np.asarray(files_name)
    return np.asarray(files), np.asarray(files_name)

def mnt_reader(file_name):
    f = open(file_name)
    ground_truth = []
    for i, line in enumerate(f):
        if i < 2 or len(line) == 0: continue
        try:
            w, h, o = [float(x) for x in line.split()]
            w, h = int(round(w)), int(round(h))
            ground_truth.append([w, h, o])
        except:
            try:
                w, h, o, _, _ = [float(x) for x in line.split()]
                w, h = int(round(w)), int(round(h))
                ground_truth.append([w, h, o])
            except:
                pass
    f.close()
    return ground_truth

def mnt_writer(mnt, image_name, image_size, file_name):
    f = open(file_name, 'w')
    f.write('%s\n'%(image_name))
    f.write('%d %d %d\n'%(mnt.shape[0], image_size[0], image_size[1]))
    for i in xrange(mnt.shape[0]):
        f.write('%d %d %.6f\n'%(mnt[i,0], mnt[i,1], mnt[i,2]))
    f.close()
    return

def load_image_mnt(image_base, mnt_base, image_ext = ".bmp"):
    image_files = glob.glob(os.path.join(image_base, "*" + image_ext))
    images, minutiae = [], []
    for i in image_files:
        _, name = os.path.split(i)
        name, ext = os.path.splitext(name)
        images.append(np.array(misc.imread(i, mode='L'), dtype=float)/255.0)
        minutiae.append(mnt_reader(mnt_base+name+'.mnt'))
    return images, minutiae

# tang sir version
def mnt2map(minutiae, w, h, batchsize):
    minutiae_map_out, minutiae_map_weight = [], []
    falg =1
    for minutia in minutiae:
        minutiae_map = np.zeros([w,h],np.float32)
        minutiae_weight = np.zeros([w,h],np.float32)
        for j in minutia:
            minutiae_map[j[1],j[0]] = 1
            minutiae_weight[max(j[1]-15,0):min(j[1]+15,w),max(j[0]-15,0):min(j[0]+15,w)] = 1
            minutiae_weight[j[1],j[0]] = 2
        fg_inds = np.where(minutiae_weight==2)
        bg_inds = np.where(minutiae_weight==0)
        fg_batch = np.random.choice(len(fg_inds[0]),min(len(fg_inds[0]),batchsize/2),replace=False)
        bg_batch = np.random.choice(len(bg_inds[0]),batchsize-len(fg_batch),replace=False)
        fg_inds = np.vstack(fg_inds)[:,fg_batch]
        bg_inds = np.vstack(bg_inds)[:,bg_batch]
        minutiae_weight = np.zeros([w,h],np.float32)
        minutiae_weight[np.ndarray.tolist(fg_inds)] = 1
        minutiae_weight[np.ndarray.tolist(bg_inds)] = 1
        minutiae_map_out.append(minutiae_map)
        minutiae_map_weight.append(minutiae_weight)
    return np.asarray(minutiae_map_out), np.asarray(minutiae_map_weight)
# gaof version
def minutiae2map(images, minutiae, scale=1):
    assert (scale % 2 == 0 or scale == 1) and scale > 0
    labels, weights = [], []
    r_len = 8 / scale
    r_weights = gaussian2d((r_len*2+1, r_len*2+1), 10.0 / scale)
    r_weights /= r_weights[r_len, r_len]
    for i in xrange(len(images)):
        target_size = np.array(images[i].shape[:2], dtype=int)/scale
        label = np.zeros(target_size+r_len*2)
        for w, h, theta in minutiae[i]:
            w, h = int((w-1) / scale)+r_len, int((h-1) / scale)+r_len
            label[h-r_len:h+r_len+1, w-r_len:w+r_len+1] = np.max(np.array([r_weights, label[h-r_len:h+r_len+1, w-r_len:w+r_len+1]]), 0)
        label = label[r_len:label.shape[0]-r_len, r_len:label.shape[1]-r_len]
        weight = 1 + label*5 # lambda for minutiae region
        labels.append(label)
        weights.append(weight)
    return labels, weights

def draw_minutiae(image, minutiae, fname, r=15):
    image = np.squeeze(image)
    fig = plt.figure()
    plt.imshow(image,cmap='gray')
    plt.hold(True)
    plt.plot(minutiae[:, 0], minutiae[:, 1], 'r.', linewidth=0.5)
    for x, y, o in minutiae:
        plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.savefig(fname)
    plt.close(fig)
    return

def draw_ori(orientation, coherence, strides):
    orientation = np.squeeze(orientation)
    ori_image = np.zeros(np.array(orientation.shape) * strides, dtype=int)
    for i in xrange(orientation.shape[0]):
        for j in xrange(orientation.shape[1]):
            cos_theta = np.cos(orientation[i, j])
            sin_theta = np.sin(orientation[i, j])
            for l in xrange(int(coherence[i, j]*(strides+1)/2)):
                x = int(cos_theta * l)
                y = int(sin_theta * l)
                ori_image[i*strides+y,j*strides+x]=255
                ori_image[i*strides-y,j*strides-x]=255
    return ori_image

def draw_ori_on_img(img, ori, mask, fname, coh=None, stride=16):
    ori = np.squeeze(ori)
    mask = np.squeeze(np.round(mask))
    img = np.squeeze(img)
    ori = ndimage.zoom(ori, np.array(img.shape)/np.array(ori.shape, dtype=float), order=0)
    if mask.shape != img.shape:
        mask = ndimage.zoom(mask, np.array(img.shape)/np.array(mask.shape, dtype=float), order=0)
    if coh is None:
        coh = np.ones_like(img)
    fig = plt.figure()
    plt.imshow(img,cmap='gray')
    plt.hold(True)  
    for i in xrange(stride,img.shape[0],stride):
        for j in xrange(stride,img.shape[1],stride):
            if mask[i, j] == 0:
                continue
            x, y, o, r = j, i, ori[i,j], coh[i,j]*(stride*0.9)
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.savefig(fname)
    plt.close(fig)            
    return

def draw_ori_out(ori_out, seg_out, output_dir):
    ori_out = np.squeeze(ori_out)
    seg_out = np.squeeze(seg_out)
    for i in xrange(ori_out.shape[0]):
        for j in xrange(ori_out.shape[1]):
            if seg_out[i,j]>0.5:
                fig = plt.figure()           
                plt.plot(np.arange(1,180,2),ori_out[i,j,:])
                plt.savefig('%s/%d_%d.png'%(output_dir,i,j))
                plt.close(fig) 

def indicator(x, name=None):
    x = tf.select(tf.greater(x,0.0), tf.ones_like(x), tf.zeros_like(x))
    return x

def gausslabel(length=180, stride=2):
    gaussian_pdf = signal.gaussian(length+1, 3)
    label = np.reshape(np.arange(stride/2, length, stride), [1,1,-1,1])
    y = np.reshape(np.arange(stride/2, length, stride), [1,1,1,-1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length-delta)+length/2
    return gaussian_pdf[delta]

def gaussian2d(shape=(5,5),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def binary2list(num, lenght=8):
    ls = [0] * lenght
    for i in xrange(lenght):
        ls[i] = (num >> i & 1)
    return ls

def num_0to1(array):
    array1 = [array[-1]] + array[:-1]
    return np.sum(np.abs(np.array(array)-np.array(array1)))/2

def list2kernal(a):
    b = [a[x] for x in [2, 1, 8, 3, 0, 7, 4, 5, 6]]
    return np.reshape(np.array(b), [3, 3])

def thinning_mask():
    kernal_1, kernal_2= [], []
    bias_1, bias_2 = [], []
    for i in xrange(256):
        kernal_i = binary2list(i)
        if 2<=sum(kernal_i)<=6 and num_0to1(kernal_i)==1:
            kernal_i.insert(0, 1)
            if kernal_i[1]*kernal_i[5]*kernal_i[7] == 0 and \
                kernal_i[3]*kernal_i[5]*kernal_i[7] == 0:
                    kernal_1.append(list2kernal(kernal_i))
            if kernal_i[1]*kernal_i[3]*kernal_i[5] == 0 and \
                kernal_i[1]*kernal_i[3]*kernal_i[7] == 0:
                    kernal_2.append(list2kernal(kernal_i))
    bias_1 = [-np.sum(x)+0.5 for x in kernal_1]
    bias_2 = [-np.sum(x)+0.5 for x in kernal_2]
    kernal_1 = np.transpose(np.reshape(np.array(kernal_1), (-1, 3, 3, 1)), [1, 2, 3, 0])
    kernal_2 = np.transpose(np.reshape(np.array(kernal_2), (-1, 3, 3, 1)), [1, 2, 3, 0])
    kernal_1[kernal_1==0] = -1
    kernal_2[kernal_2==0] = -1
    return kernal_1, kernal_2, bias_1, bias_2

def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3
    xmax = ksize[0]/2
    ymax = ksize[1]/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def gabor_bank(stride=2):
    filters = np.ones([25,25,180/stride], dtype=float)
    for n, i in enumerate(xrange(-90,90,stride)):  
        theta = i*np.pi/180.
        kernel = gabor_fn((24,24),4.5, -theta, 14, 0, 0.5)
        filters[..., n] = kernel
    filters = np.reshape(filters,[25,25,1,-1])
    return filters

def large_data_size(data):
    return data.get_shape()[1] > 1 and data.get_shape()[2] > 1


def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta
def fmeasure(P, R):
    return 2*P*R/(P+R+1e-10)
def distance(y_true, y_pred, max_D=16, max_O=np.pi/6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    O = spatial.distance.cdist(np.reshape(y_true[:, 2], [-1, 1]), np.reshape(y_pred[:, 2], [-1, 1]), angle_delta)
    return (D<=max_D)*(O<=max_O)
def mnt_P_R_F(y_true, y_pred):
    if y_pred.shape[0]==0 or y_true.shape[0]==0:
        return 0, 0, 0, 0, 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    total_gt, total = float(y_true.shape[0]), float(y_pred.shape[0])
    matched = distance(y_true, y_pred)
    recall = np.max(matched, axis=-1)
    recall = np.sum(recall) / total_gt
    precision_only = np.max(matched, axis=1) # maybe wrong when one pred minutiae activated two true minutiae
    precision_only = np.sum(precision_only) / total
    precision = np.max(matched, axis=0)
    precision = np.sum(precision) / total
    precision_only = np.minimum(precision, precision_only)
    return precision_only, precision, recall, fmeasure(precision_only, recall), fmeasure(precision, recall)

def nms(mnt):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in xrange(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]
        
