
from glob import glob
from urllib.request import urlretrieve
import re
import random
import zipfile
import tarfile


import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
import humandetection
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import cv2
import winsound
import imageio
sys.path.append("..")
import six.moves.urllib as urllib
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

from ShadowRemoval import ShadowRemover as sr
reader = imageio.get_reader('./video/Untitled_7_kisakisa.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('./video/UNTITLED_7_v3_.mp4', fps=fps)

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Training Finished!')
    print('Saving test images to: {}, please wait...'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    print('All augmented images are saved!')


def gen_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):

    for i, input_image in enumerate(reader):

        image = scipy.misc.imresize(input_image, image_shape)

        startTime = time.clock()
        
        image_file = './runs/a.png'
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")

        scipy.misc.imsave("mask.png", mask)
   
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
 
        
        last_image = humandetection.findHuman(np.array(image))
        
        scipy.misc.imsave("last_image.jpg", last_image)

        namestreet= 'street_im' + str(i) + '.png'
        scipy.misc.imsave(namestreet, street_im)
        



        I1 = np.array(mask) 
        I1 = I1[:, :, ::-1].copy() #yol segmenti

        I2 = np.array(last_image) 
        I2 = I2[:, :, ::-1].copy()  #yeşilleri mask edicez (person çerçeve rengi = chartreuse)
        
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)


        name4 = 'I1'+ str(i)+ '.png'
        scipy.misc.imsave( name4, I1)
        name5 = 'I2'+ str(i)+ '.png'
        scipy.misc.imsave(name5, I2)
        I1[I1 <91 ] = 0
        I1[I1 >91 ] = 0
        I1[I1 ==91 ] = 255 #mask image siyah beyaza dönüştürüldü
        I1 = np.uint8(I1)

        namemask = 'MASKsiyahbeyaz' + str(i) + '.png'
        scipy.misc.imsave(namemask, I1)
        
        rgb_cerceve=cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)
        scipy.misc.imsave("I2_before.jpg", rgb_cerceve)
#        plt.imshow(I2)
#        plt.show()
        hsv_cerceve=cv2.cvtColor(rgb_cerceve,cv2.COLOR_RGB2HSV)
        scipy.misc.imsave("I2_after.jpg", hsv_cerceve)

        cerceverengi_low = (45,255,255)
        cerceverengi_high = (45,255,255) #chartreuse renginin rgb(127,255,0) kodunun rgbtohsvden sonraki renk kodu bu

        cercevemask = cv2.inRange(hsv_cerceve,cerceverengi_low,cerceverengi_high)
        
        result = cv2.bitwise_and(rgb_cerceve,rgb_cerceve,mask=cercevemask)

        #BU cerceve mask şuan yeşilleri seçti ve bunu siyahbeyaza çevirmeliyiz.

        name0 = 'cerceve'+ str(i)+ '.png'
        scipy.misc.imsave(name0, result)

        bwor_entegrasyon = cv2.bitwise_or(np.array(result), np.array(street_im))  # human ayrı detect edilmiş halive yolun ayrı segmente edilmiş hali birleştirilerek çıktıya verilir

        namebwor = 'bwor' + str(i) + '.png'
        scipy.misc.imsave(namebwor, bwor_entegrasyon)

        
        gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        gray_result[gray_result > 179] = 255     #gri yerler beyaz yapıldı
        
        gray_result = np.uint8(gray_result)

        name1 = 'out'+ str(i)+ '.png'
        scipy.misc.imsave(name1, gray_result)
       
        
        bwa_KESISIM = cv2.bitwise_and(gray_result, I1)
        

        name2 = 'I4'+ str(i)+ '.png'
        scipy.misc.imsave(name2, bwa_KESISIM)
        
        bwa_KESISIM[ (bwa_KESISIM>200) ] = 255 

        n2 = np.sum(bwa_KESISIM == 255)

        print("Kesişim Toplamı: ", n2 )
        if(n2>0):
            print('Human in the road!!!!!!!!')
            winsound.PlaySound('C:/tensorflow1/models/research/object_detection/video/alarm.wav', winsound.SND_LOOP)
            # plt.imshow(bwor_entegrasyon)
            # plt.show(block=False)
            # plt.pause(5)
            # plt.close()

        # plt.imshow(bwor_entegrasyon)
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()


        writer.append_data(bwor_entegrasyon)
        print('writing %d',i)
        print(i)

    writer.close()
    
    endTime = time.clock()
    speed_ = 1.0 / (endTime - startTime)

    yield os.path.basename(image_file), np.array(last_image), speed_

def pred_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Predicting images...')
    # start epoch training timer
    image_outputs = gen_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)

    counter = 0
    for name, image, speed_ in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        if print_speed is True:
            counter+=1
            print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

        # sum_time += laptime

    # pngCounter = len(glob1(data_dir,'*.png'))

    print('All augmented images are saved to: {}.'.format(output_dir))
