import argparse
import pandas as pd

from model import segnet
from generator import data_gen_small


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(
            description="SegNet LIP dataset")
    parser.add_argument("--save_dir",
            help="output directory", default='./')
    parser.add_argument("--train_list",
            help="train list path")
    parser.add_argument("--trainimg_dir",
            help="train image dir path")
    parser.add_argument("--trainmsk_dir",
            help="train mask dir path")
    parser.add_argument("--val_list",
            help="val list path")
    parser.add_argument("--valimg_dir",
            help="val image dir path")
    parser.add_argument("--valmsk_dir",
            help="val mask dir path")
    parser.add_argument("--batch_size", default=28, type=int,
            help="batch size")
    parser.add_argument("--n_epochs", default=50, type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps", default=4500, type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps", default=10, type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels", default=2, type=int,
            help="Number of label")
    parser.add_argument("--input_shape", default=(400, 400, 3),
            help="Input images shape")
    parser.add_argument("--kernel", default=3, type=int,
            help="Kernel size")
    parser.add_argument("--pool_size", default=(2, 2),
            help="pooling and unpooling size")
    parser.add_argument("--output_mode", default="softmax", type=str,
            help="output activation")
    parser.add_argument("--loss", default="categorical_crossentropy", type=str,
            help="loss function")
    parser.add_argument("--optimizer", default="adadelta", type=str,
            help="optimizer")
    args = parser.parse_args()

    return args

import os
def read_files_recursive(path, fileNamesOnly=False, writeToFile='test.txt'):
    '''Read filenames in the path. '''
    filelist = []
    for item in os.listdir(path):
       # ident, sess, num = item.split('_')
       # ids.append(int(ident))
        if os.path.isdir(os.path.join(path, item)):

            subfiles = read_files_recursive(os.path.join(path, item), writeToFile=None)

            [filelist.append(f) for f in subfiles]

            #filelist.append(item)
        elif not os.path.isdir(os.path.join(path, item)) and item.endswith('.png'):
            if fileNamesOnly:
                filelist.append(item)
            else:
                filelist.append(os.path.join(path, item))
        #os.path.join(path,item)

    if writeToFile:
        with open(writeToFile, 'w') as f:
            for item in filelist:
                    f.write("%s\n" % item)

    return filelist

def load_kumara(kumara_path, loading_fun, *loading_args, **loading_kwargs):
    "General way of loading, preloading and storing pickles."
    import pickle

    if os.path.isfile(kumara_path): 
        input('Kumara FOUND, preload will happen. Confirm. ') 
        # Getting back the objects: 
        with open(kumara_path, 'rb') as f: 
            kumara = pickle.load(f) 
    else: 
        input('Kumara NOT found, regular loading will begin. Confirm. ') 
        # Load object as would regularly do.
        kumara = loading_fun(*loading_args, **loading_kwargs)
        # Save it for faster preloading next time.
        with open(kumara_path, 'wb') as handle:
            pickle.dump(kumara, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return kumara

def main(args):

    # gpu selection
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

	sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	from utils import get_eyez_dir
		
    src_dir = os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'train')
    val_dir = os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'val')

    import pickle
    train_list_pickle = './pickles/train_list.kumara'    
    l = load_kumara(train_list_pickle, read_files_recursive, path=(src_dir+'SegNet_Image'), fileNamesOnly=True)

    # if os.path.isfile(train_list_pickle): 
    #     # Getting back the objects: 
    #     input('Pickle FOUND, preload will happen. Confirm. ') 
    #     with open(train_list_pickle, 'rb') as f: 
    #         l = pickle.load(f) 
    # else: 
    #     input('Pickle not found, image loading will begin. Confirm. ') 
    #     # TUKI KODA ZA LOADANJE
    #     l = read_files_recursive(src_dir+'SegNet_Image', fileNamesOnly=True)
    #     with open(train_list_pickle, 'wb') as handle:
    #         pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_list_pickle = './pickles/test_list.kumara'
    val_list = load_kumara(test_list_pickle, read_files_recursive, path=(val_dir+'SegNet_Test_Image/'), fileNamesOnly=True)

    # if os.path.isfile(test_list_pickle): 
    #     # Getting back the objects: 
    #     input('Pickle FOUND, preload will happen. Confirm. ') 
    #     with open(test_list_pickle, 'rb') as f: 
    #         val_list = pickle.load(f) 
    # else: 
    #     input('Pickle not found, image loading will begin. Confirm. ') 
    #     # TUKI KODA ZA LOADANJE
    #     #l = read_files_recursive(src_dir+'SegNet_Image', fileNamesOnly=True)
    #     val_list = read_files_recursive(val_dir + 'SegNet_Test_Image/', fileNamesOnly=True)
    #     # save to pickle
    #     with open(test_list_pickle, 'wb') as handle:
    #         pickle.dump(val_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #exit()

    # split into train/test set
    import random
    random.shuffle(l)
    random.shuffle(val_list)
    
    #split_ix = int(len(l)*0.7)

    train_list = l  # l[0:split_ix]
    val_list = val_list[0:20000] #l[split_ix:]
    
    train_list = pd.DataFrame(train_list)
    val_list = pd.DataFrame(val_list)
#    print(l[:10])
#    exit()

    # set the necessary list
    #train_list = pd.read_csv(args.train_list, header=None)
    #val_list = pd.read_csv(args.val_list, header=None)

    # set the necessary directories
    trainimg_dir = os.path.join(src_dir, 'Images') # args.trainimg_dir
    trainmsk_dir = os.path.join(src_dir, 'Masks') #args.trainmsk_dir
    valimg_dir = os.path.join(val_dir, 'Images') #args.valimg_dir
    valmsk_dir = os.path.join(val_dir, 'Masks') #args.valmsk_dir


    train_gen = data_gen_small(trainimg_dir, trainmsk_dir,
            train_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)
    val_gen = data_gen_small(valimg_dir, valmsk_dir,
            val_list, args.batch_size,
            [args.input_shape[0], args.input_shape[1]], args.n_labels)

    model = segnet(args.input_shape, args.n_labels,
            args.kernel, args.pool_size, args.output_mode)
    print(model.summary())

    ### calculate class weights
    ### TODO!
    #from sklearn.utils import class_weight
    #class_weights = class_weight.compute_class_weight('balanced',
    #                                             np.unique(y_train),
    #                                             y_train)
    # THEN add as: , class_weight=class_weights

    # on positive examples
    #class_weight = {0: 0.750162010773949, 1: 0.2498379892260511}

    model.compile(loss=args.loss,
            optimizer=args.optimizer, metrics=["accuracy"])
    model.fit_generator(train_gen, steps_per_epoch=args.epoch_steps,
            epochs=args.n_epochs, validation_data=val_gen,
            validation_steps=args.val_steps)

    model.save(args.save_dir+"SegNet_model.e"+str(args.n_epochs)+".bs"+str(args.batch_size)+".h5")
    model.save_weights(args.save_dir+"SegNet_weights.e"+str(args.n_epochs)+".hdf5")
    print("saving weights done..")


if __name__ == "__main__":
    args = argparser()
    main(args)
