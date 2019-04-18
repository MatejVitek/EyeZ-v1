

### calculate class weights
    ### TODO!
import numpy as np
from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread


if __name__ == '__main__':

    from train import read_files_recursive, argparser
    
    args = argparser()
    dims = args.input_shape
    
    src_dir = './data/SegNet_trainset2/SegNet_trainset/'
    # read files
    l = read_files_recursive(src_dir+'SegNet_Label', fileNamesOnly=True)

    img_path = src_dir + 'SegNet_Label/' + l[0]
    original_img = imread(img_path, mode='RGB')
    resized_img = imresize(original_img, (dims[0], dims[1])) # +[3]


    def unique_category_label(labels, dims, n_labels):


        #old_shape = labels.shape
        #flat_labels = labels.flatten()

        uniq_vals, indexes = np.unique(labels, return_index=True)
        d = {}
        for ix, lab in enumerate(uniq_vals):
            d[lab] = 0

        #    label_indices = (labels == lab)
        #    labels[label_indices] = ix

        #labels = flat_labels.reshape(old_shape)

        

        #x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                #if (labels[i][j] == 0): # skip background
                #    continue
                d[labels[i][j]] += 1
                #x[i, j, labels[i][j]] = 1
        #if n_labels == 1:
        #    x = x.reshape(dims[0] * dims[1])
        #else:
        #x = x.reshape(dims[0] * dims[1], n_labels)
        return d#x

    d_total = {}
    y_train = np.zeros((len(l), dims[0]* dims[1]))
    for ix, img in enumerate(l):
        print(ix)
        img_path = src_dir + 'SegNet_Label/' + img
        original_img = imread(img_path, mode='L')
        resized_img = imresize(original_img, (dims[0], dims[1]), interp='nearest') # +[3]
        
        if ix == 0:
            d_total = unique_category_label(resized_img, dims, 2)
        else:
            d = unique_category_label(resized_img, dims, 2)
            for key in d:
                d_total[key] += d[key]

        #y_train[ix, :] = reshaped_img

    #print("y_train.shape: ", y_train.shape)
    
    #from sklearn.utils import class_weight
    #class_weights = class_weight.compute_class_weight('balanced',
    #                                            np.unique(y_train),
    #                                            y_train)

    total_pixels = float(len(l) * dims[0] * dims[0])
    for key in d_total.keys():
        d_total[key] = d_total[key] / total_pixels

    print("Class weights: ", d_total)

    # THEN add as: , class_weight=class_weights
