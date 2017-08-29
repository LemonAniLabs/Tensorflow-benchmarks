import skimage
import skimage.io
import numpy as np
import os,sys
import tensorflow as tf
import time
sys.path.append('/home/lennon.lin/Repository/github/tensorflow-Model/slim')
try:
    import urllib2
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import resnet_v1,vgg
from preprocessing import vgg_preprocessing

from tensorflow.contrib import slim

image_size = vgg.vgg_16.default_image_size
checkpoints_dir='/home/lennon.lin/Checkpoint/'

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #resized_img = skimage.transform.resize(crop_img, (size, size))
    #img = img.reshape((224,224,3), dtype=np.float32)
    #img = np.array(img, dtype=np.float32)
    img = img.reshape(1, 224, 224, 3)
    return img

#with tf.Graph().as_default():
url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg'
image_string = urllib.urlopen(url).read()
image = tf.image.decode_jpeg(image_string, channels=3)
images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
img=load_image('1.jpg',224)
print('type of image -> ', img)
processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # 1000 classes instead of 1001.
    logits, _ = resnet_v1.resnet_v1_50(images, num_classes=1000, is_training=False)
probabilities = tf.nn.softmax(logits)
    
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
    slim.get_model_variables('resnet_v1_50'))
#sess = tf.InteractiveSession() 
#sess = tf.Session()
with tf.Session() as sess:
    init_fn(sess)
    #sess.run(tf.global_variables_initializer())
    #        probabilities = sess.run(probabilities, feed_dict={images:img})
    total_time=0
    for i in range(20):
        begin = time.time()
        pred = sess.run(probabilities,feed_dict={images:img})
        end = time.time()
    
        once_time = end-begin
        print('time->', once_time)
        total_time +=once_time
    
    print('mean time -> ', total_time/20)
    pred = pred[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-pred), key=lambda x:x[1])]
        #plt.figure()
        #plt.imshow(np_image.astype(np.uint8))
        #plt.axis('off')
        #plt.show()
        
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
            # Shift the index of a class name by one. 
        print('Probability %0.2f%% => [%s]' % (pred[index] * 100, names[index+1]))
