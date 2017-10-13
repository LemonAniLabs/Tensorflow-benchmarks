import cv2
import numpy as np
import argparse
import os,sys
import tensorflow as tf
import time
from datasets import imagenet
from tensorflow.contrib import slim

def parseArg():

    parser = argparse.ArgumentParser(description='Tensorflow Resnet_v1 Benchmark')
#    parser.add_argument('-i', '--IMG_FILE', default=None, help='Image URL Path')
#    parser.add_argument('-u', '--IMG_URL', default=None, help='Image URL Path')
    parser.add_argument('-t', '--TIMES', default=100, help='Times of iteration', type=int)
    parser.add_argument('-o', '--OFFICIAL', action='store_true', help='Use Slim.Net from tensorflow contrib')
    args = parser.parse_args()
    print(args)
    return args


def load_image(path, size=224):
    img = cv2.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #resized_img = skimage.transform.resize(crop_img, (size, size))
    #img = img.reshape((224,224,3), dtype=np.float32)
    #img = np.array(img, dtype=np.float32)
    img = img.reshape(1, 224, 224, 3)
    return img

def main(args):
    _args = parseArg()
    
    if _args.OFFICIAL:
        from tensorflow.contrib.slim.nets import resnet_v1, vgg
        pred_shape=[]
    else:
        from nets import resnet_v1,vgg
    
    image_size = vgg.vgg_16.default_image_size
    checkpoints_dir='/home/lennon.lin/Checkpoint/'
    
    images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
    img=load_image('1.jpg',224)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # 1000 classes instead of 1001.
        logits, _ = resnet_v1.resnet_v1_50(images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)
        
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
        slim.get_model_variables('resnet_v1_50'))
    
    with tf.Session() as sess:
        
        init_fn(sess)
        total_time=0
        
        # Skip the first time
        sess.run(probabilities,feed_dict={images:img})
        for i in range(_args.TIMES):
            begin = time.time()
            pred = sess.run(probabilities,feed_dict={images:img})
            end = time.time()
        
            once_time = end-begin
            print('time-> %.8f' % once_time)
            total_time +=once_time
        
        print('mean time -> %.8f ' % (total_time/_args.TIMES))
        
        pred = np.squeeze(pred)
        sorted_inds = [i[0] for i in sorted(enumerate(-pred), key=lambda x:x[1])]
            
        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            # Shift the index of a class name by one.
            print('Probability %0.2f%% => [%s]' % (pred[index] * 100, names[index+1]))
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
