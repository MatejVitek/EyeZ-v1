import argparse

from keras.models import load_model

from keras import backend as K
import numpy as np

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

from model import segnet
from train import read_files_recursive

SIGMA = 1.5
MODE = 1

def generateGaussianMask(size, isRGB=True, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is stddev (effective radius).
    """
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2 # integer division (Py.3 uses //)
    else:
        x0 = center[0]
        y0 = center[1]
    
    sigma = size / 5.0 # before size / 3.0 when 2 was not included in Gaussian
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (SIGMA*sigma)**2)
    if isRGB:
        mask = np.stack((mask,)*3, axis=-1)

    #plt.imshow(mask)
    #plt.show()
    return mask


def predict_n_join(src_dir, src_pattern, out_dir, isRGB=True, targetSize=5000):

    gaussian_mask = generateGaussianMask(41, isRGB=True)
    dict = {}
    dictW = {}

    l = read_files_recursive(test_dir, fileNamesOnly=True)
    l = [i for i in l if src_pattern in i]

    def reshape_n_resize(image):
        image = image.reshape((args.input_shape[0], args.input_shape[1]))
        image = imresize(image, (h, w, 1), interp='nearest')
        return image


    for ix, img_name in enumerate(l):
        
        if (ix % 10000 == 0):
            print(ix, " : ", img_name)
        img_path = os.path.join(test_dir, img_name)
        image = imread(img_path, mode='RGB')#[:, :, ::-1]
        
        h,w,ch = image.shape
        image = imresize(image, (args.input_shape[0], args.input_shape[1])) / 255.0 #  interp='nearest'

        #from scipy.misc import toimage
        #toimage(image).show()

        gen = model.predict(np.array([image]))[0]#[:,1]

        back = gen[:,0]
        mask = gen[:,1]

        back = reshape_n_resize(back)
        mask = reshape_n_resize(mask)    

        #from scipy.misc import toimage
        #toimage(mask).show()

        #dest_path = './data/SegNet_trainset/my_test/predicted_rot180.png'
        #imsave(os.path.join(dest_path, item), img)
        #imsave(os.path.join(test_dir, 'back_180.png'), back)

        #    imsave(os.path.join(out_dir, img_name), mask)
        pArr = img_name.split('_')
        id = str(pArr[0])
        x = int(pArr[1][1:])
        y = int(pArr[2][1:])
        
        if not id in dict:
            dict[id] = np.zeros([targetSize,targetSize,c],dtype=np.int32)
            dictW[id] = np.zeros([targetSize,targetSize,c],dtype=float)

        if DEBUG:
            # break the image - for the doggo experiment, so that we see where the patches are:
            mask = mask * np.random.rand(1,1,3)

        if MODE == 1 or MODE == -1:
            mask = mask * gaussian_mask
        if MODE == 1:
            
            dict[id][y:y+h, x:x+w, :] = dict[id][y:y+h, x:x+w, :] + mask
            dictW[id][y:y+h, x:x+w, :] = dictW[id][y:y+h, x:x+w, :] + gaussian_mask
        else:
            dict[id][y:y+h, x:x+w, :] = img

        if not isRGB:
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for key, img in dict.items():
        if MODE == 1:
            img = img / dictW[key]
            if DEBUG:
                img = img * SIGMA
        imsave(os.path.join(out_dir, src_pattern + "_" + str(key) + ".png" ), img)
        #cv2.imwrite(out_path, img)


def main(args):
    
    # gpu selection
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

    model_path = './'
    #model_path = os.path.join(model_path, 'SegNet.e10.bs28.hdf5')
    model_path = os.path.join(model_path, 'seg.hdf5')
    
    print("MODEL path: ", model_path)
    #input("Press any key to continue...")
    try:
        model = load_model(model_path)
    except ValueError: # No model found in config file.
        print("No model found in config. Generating model and loading weights!")
        model = segnet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
        model.load_weights(model_path)
    
    model.summary()



    test_dir = './data/SegNet_trainset/SegNet_Image/'
    out_dir = './data/SegNet_trainset/SegNet_Predicted_Train/'
    # read files
    l = read_files_recursive(test_dir, fileNamesOnly=True) #[1:200]

    #l = [i for i in l if "11ska445800" in i]

    print("Total images: ", len(l))
    input("Press any key to continue... ")

    def reshape_n_resize(image):
        image = image.reshape((args.input_shape[0], args.input_shape[1]))
        image = imresize(image, (h, w, 1), interp='nearest')
        return image


    for ix, img_name in enumerate(l):
        
        if (ix % 10000 == 0):
            print(ix, " : ", img_name)
        img_path = os.path.join(test_dir, img_name)
        image = imread(img_path, mode='RGB')#[:, :, ::-1]

        #print(args.input_shape)
        #exit()
        #args.input_shape = (1024, 1024, 3)

        h,w,ch = image.shape
        image = imresize(image, (args.input_shape[0], args.input_shape[1])) / 255.0 #  interp='nearest'
        #image = image / 255.0

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
   
        back = reshape_n_resize(back)
        mask = reshape_n_resize(mask)    

        #from scipy.misc import toimage
        #toimage(mask).show()

        #dest_path = './data/SegNet_trainset/my_test/predicted_rot180.png'
        #imsave(os.path.join(dest_path, item), img)
        #imsave(os.path.join(test_dir, 'back_180.png'), back)

        imsave(os.path.join(out_dir, img_name), mask)

if __name__ == "__main__":
    import argparse
    from train import argparser

    args = argparser()
    main(args)

    
