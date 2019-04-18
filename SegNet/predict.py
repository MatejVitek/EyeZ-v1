import argparse

from keras.models import load_model

from keras import backend as K
import numpy as np

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

from model import segnet

def main(args):
    
    # gpu selection
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

    test_dir = './data/SegNet_trainset/my_test/'
    image = imread(os.path.join(test_dir, 'largest.png'), mode='RGB')#[:, :, ::-1]

    #print(args.input_shape)
    #exit()
    args.input_shape = (1024, 1024, 3)

    h,w,ch = image.shape
    image = imresize(image, (args.input_shape[0], args.input_shape[1])) / 255.0 #  interp='nearest'
    #image = image / 255.0

    out_path = './'
    model_path = os.path.join(out_path, 'SegNet.e10.bs28.hdf5')
    model_path = os.path.join(out_path, 'seg.hdf5')
    
    print("MODEL path: ", model_path)
    #input("Press any key to continue...")
    try:
        model = load_model(model_path)
    except ValueError: # No model found in config file.
        print("No model found in config. Generating model and loading weights!")
        model = segnet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
        model.load_weights(model_path)
    
    model.summary()

    
    #from scipy.misc import toimage
    #toimage(image).show()

    gen = model.predict(np.array([image]))[0]#[:,1]

    back = gen[:,0]
    mask = gen[:,1]

    #if K.image_dim_ordering() == 'th':
    #    image = np.empty(gen.shape[2:] + (3,))
    #    for x in range(0, 3):
    #        image[:, :, x] = gen[x, :, :]
    #else:
    #    image = gen
    #image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)

    def reshape_n_resize(image):
        image = image.reshape((args.input_shape[0], args.input_shape[1]))
        image = imresize(image, (h, w, 1), interp='nearest')
        return image

    back = reshape_n_resize(back)
    mask = reshape_n_resize(mask)    

    #from scipy.misc import toimage
    #toimage(mask).show()

    #dest_path = './data/SegNet_trainset/my_test/predicted_rot180.png'
    #imsave(os.path.join(dest_path, item), img)
    imsave(os.path.join(test_dir, 'back_180.png'), back)
    imsave(os.path.join(test_dir, 'mask_180.png'), mask)

if __name__ == "__main__":
    import argparse
    from train import argparser

    args = argparser()
    main(args)

    
