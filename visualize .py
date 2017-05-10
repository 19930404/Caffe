#url http://hirotaka-hachiya.hatenablog.com/entry/2015/02/15/221804
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
caffe_root = "/home/kobayashi/caffe/"
sys.path.insert(0, caffe_root + 'python')
import caffe

# function for visualizing filters
def plot_images(images, tile_shape):
    assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  nrows_ncols = tile_shape ) 
    for i in range(images.shape[0]):
        grd = grid[i]
        grd.imshow(images[i])

# set the pathes of model definition, trained model and image file to be predicted
MODEL_FILE = caffe_root + '50-50/apple_width_shift_range/deploy.prototxt'
PRETRAINED = caffe_root + '50-50/apple_width_shift_range/snapshot_iter_13546.caffemodel'
LABEL_FILE = caffe_root + 'data/ilsvrc12/words.txt'
MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
IMAGE_FILE = '/home/kobayashi/images/blackrotmiss.jpg'

# read labels of classes
with open(LABEL_FILE) as f:
    labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])

# load classifier
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# load image file to be predicted
input_image = caffe.io.load_image(IMAGE_FILE)

# predict
prediction = net.predict([input_image])
sorted_predict = sorted(range(len(prediction[0])),key=lambda x:prediction[0][x],reverse=True)

print 'Results:'
# print top5 result
'''
rank=1
for i in sorted_predict[0:5]:
        print 'Top',rank,labels_df['name'][i],'(',i,')',', score=',prediction[0][i]
	rank +=1
'''
# plot image and result
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(input_image)
plt.subplot(2,1,2)
#plt.plot(prediction[0])

# visualize filters
plt.figure(2)
plot_images(net.blobs['conv1/7x7_s2'].data[0],tile_shape=(8,8))
plt.show()
