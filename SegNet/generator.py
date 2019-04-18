import cv2
import numpy as np

from keras.preprocessing.image import img_to_array

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x

def unique_category_label(labels, dims, n_labels):

    #old_shape = labels.shape
    #flat_labels = labels.flatten()

    uniq_vals, indexes = np.unique(labels, return_index=True)

    for ix, lab in enumerate(uniq_vals):
        label_indices = (labels == lab)
        labels[label_indices] = ix

    #labels = flat_labels.reshape(old_shape)

    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            #if (labels[i][j] == 0): # skip background
            #    continue
            x[i, j, labels[i][j]] = 1
    #if n_labels == 1:
    #    x = x.reshape(dims[0] * dims[1])
    #else:
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            img_path = img_dir + lists.iloc[i, 0] #+ '.jpg'
            #original_img = cv2.imread(img_path)[:, :, ::-1]
            #resized_img = cv2.resize(original_img, (dims[0], dims[1])) # +[3]
            original_img = imread(img_path, mode='RGB')#[:, :, ::-1]
            resized_img = imresize(original_img, (dims[0], dims[1])) # +[3]

            array_img = img_to_array(resized_img)/255.0
            imgs.append(array_img)
            # masks
            #original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] ) # + '.png'
            #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
            #array_mask = unique_category_label(resized_mask[:, :, 0], dims, n_labels)
            original_mask = imread(mask_dir + lists.iloc[i, 0], mode='L')
            resized_mask = imresize(original_mask, (dims[0], dims[1]), interp='nearest') # +[3]
            #array_mask = category_label(resized_mask[:, :, 0], dims, n_labels)

            array_mask = unique_category_label(resized_mask, dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels




if __name__ == '__main__':

    from train import read_files_recursive
    import pandas as pd
    import argparse
    from train import argparser

    args = argparser()

    # gpu selection
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo


    # read files
    l = read_files_recursive('./data/SegNet_trainset/SegNet_Image', fileNamesOnly=True)

    # split into train/test set
    import random
    random.shuffle(l)
    split_ix = int(len(l)*0.7)

    train_list = l[0:split_ix]
    val_list = l[split_ix:]
    
    train_list = pd.DataFrame(train_list)
    val_list = pd.DataFrame(val_list)

    # set the necessary directories
    trainimg_dir = './data/SegNet_trainset/SegNet_Image/' # args.trainimg_dir
    trainmsk_dir = './data/SegNet_trainset/SegNet_Label/' #args.trainmsk_dir
    valimg_dir = './data/SegNet_trainset/SegNet_Image/' #args.valimg_dir
    valmsk_dir = './data/SegNet_trainset/SegNet_Label/' #args.valmsk_dir

    train_gen = data_gen_small(trainimg_dir, trainmsk_dir,
            train_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)
    val_gen = data_gen_small(valimg_dir, valmsk_dir,
            val_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)

    i = 0
    for x, y in train_gen:

        imsave("./debug/img:"+str(i)+".png", x[0])
        imsave("./debug/mask:"+str(i)+".png", y[0])

        i=i+1

        if i%100 == 0:
            break