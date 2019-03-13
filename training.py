# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:50:05 2018

@author: emily
"""

from __future__ import division
import sys
import os
import siamese as siam
import trainer
import read_training_dataset 
import os


"""
    training procedure:
    1,input z, x, pos_x, pos_y, w, d and gt of x
    2,pad and crop z,x
    3,calculate score map
    4,calculate loss
    5,bp, update variable
"""


def train():

    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = 255

    
    # build the computational graph of Siamese fully-convolutional network
    siamNet = siam.Siamese(8)
    # get tensors that will be used during training
    image, z_crops, x_crops, templates_z, scores, loss, train_step, distance_to_gt, summary, templates_x, net_cat = siamNet.build_tracking_graph_train(final_score_sz)
 
    # read tfrecodfile holding all the training data
    data_reader = read_training_dataset.myReader(700, 700, 3)
    batched_data = data_reader.read_tfrecord(os.path.join("tfrecords", "training_dataset"), num_epochs = 50, batch_size = 8)
    
    # run trainer
    trainer.trainer(final_score_sz, batched_data, image, templates_z, scores, loss, train_step, distance_to_gt,  z_crops, x_crops, siamNet, summary,templates_x,net_cat)




if __name__ == '__main__':
#    sys.exit(train())
    train()
    
