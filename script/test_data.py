import numpy as np
import json
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/vdl_img_vgg.h5', help='path to image feature, now hdf5 file')

opt = parser.parse_args()
path = opt.path

f = h5py.File(path, 'r')
train_feats = f['images_train']
val_feats = f['images_val']

idx_list = [0,5,50,4335,2134]

actual_train_ans = [2.7524679,1.5914612,2.9992323,2.7822866,2.0583184]
actual_val_ans = [2.4914272,3.1998599,1.7559544,1.7371163,1.766684]

print('training features...')
idx = 0
for i in idx_list:
    print('your ans vs actual ans {} | {}'.format(np.mean(train_feats[i,:,:,:]), actual_train_ans[idx]))
    idx+=1

idx = 0
print('validation features...')
for i in idx_list:
    print('your ans vs actual ans:{} | {}'.format(np.mean(val_feats[i, :, :, :]), actual_val_ans[idx]))
    idx+=1
    # print(np.mean(val_feats[i,:,:,:]))
