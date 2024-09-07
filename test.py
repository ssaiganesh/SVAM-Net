import os
import cv2
import time
import ntpath
import argparse
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, exists
# keras libs
from keras.layers import Input
from keras.models import Model
# local libs
from models.svam_model import SVAM_Net
from utils.data_utils import preprocess, deprocess_mask

def sigmoid(x):
    """ Numerically stable sigmoid """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x))
                   )

def deprocess_gens(saml, sambu, samtd, out, im_res):
    """ Post-process all outputs """
    samtd, sambu =  sigmoid(samtd), sigmoid(sambu)
    out = deprocess_mask(out).reshape(im_res) 
    saml = deprocess_mask(saml).reshape(im_res) 
    samtd = deprocess_mask(samtd).reshape(im_res) 
    sambu = deprocess_mask(sambu).reshape(im_res)
    return saml, sambu, samtd, out

# input and output data shape
im_res, chans = (256, 256), 3
im_shape = (256, 256, 3)
x_in = Input(batch_shape=(1, im_res[1], im_res[0], chans))

def test_single_image(img_path, res_dir, model_h5):
    """ Test a single image """
    # Create directory for output test data if it does not exist
    if not exists(res_dir): os.makedirs(res_dir)

    # Load specific model
    assert os.path.exists(model_h5), "h5 model not found"
    model = SVAM_Net(res=im_shape)
    model.load_weights(model_h5)

    # Prepare data
    img_name = ntpath.basename(img_path).split('.')[0]
    inp_img = np.array(Image.open(img_path).resize(im_res))
    im = np.expand_dims(preprocess(inp_img), axis=0)        

    # Generate saliency
    t0 = time.time()
    saml, sambu, samd, out = model.predict(im)
    elapsed_time = time.time() - t0

    _, out_bu, _, out_tdr = deprocess_gens(saml, sambu, samd, out, im_res)
    print(f"Processed: {img_path} in {elapsed_time:.2f} sec")

    # Save images
    Image.fromarray(inp_img).save(join(res_dir, img_name + ".jpg"))
    Image.fromarray(out_bu).save(join(res_dir, img_name + "_bu.png"))
    Image.fromarray(out_tdr).save(join(res_dir, img_name + "_tdr.png"))

    print(f"Saved generated images in {res_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="data/test/sample_image.jpg")
parser.add_argument("--res_dir", type=str, default="data/output_svam/")
parser.add_argument("--ckpt_h5", type=str, default="checkpoints/SVAM_Net.h5")
args = parser.parse_args()