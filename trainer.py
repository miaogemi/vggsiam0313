import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import vgg19
#final_score_sz：255
#batched_data
#image：=shape=(8, ?, ?, 3)image原图像大小700, 
#templates_z：[24, 17, 17, 32], scores：#[24, 255, 255, 1], 
#loss = tf.reduce_mean(tf.log(1 + tf.exp(-score * self.label))), 
#train_step= tf.train.AdamOptimizer(1e-6).minimize(loss), 
#distance_to_gt是一个数 为了可视化训练过程中的距离值而已，没有真实含义, 
#summary= tf.summary.scalar('distance_to_gt', distance_to_gt)#可视化训练过程
def trainer(final_score_sz, batched_data, image, templates_z, scores, loss, train_step, distance_to_gt, z_crops, x_crops, siamNet, summary,templates_x,net_cat):
    """
        run the training steps under tensorflow session.
        
        Inputs:
            hp, run, design: system parameters.
            
            final_score_sz: size of the final score map after bilinear interpolation.
            
            batched_data: list of batched training data, consist of : z, x, z_pos_x, 
                    z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h
            
            image, templates_z, scores, loss, train_step, distance_to_gt, z_crops, 
            x_crops: tensors that will be run in tensorflow session. See siamese.py 
                     for detailed explanation.
           
            siamNet: an instance of siamese network class.
            summary: summary tensor for tensorboard.
            
        Returns:
       
    """

    # unpack data tensor from tfrecord
    z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = batched_data
    
    # create saver to log check point
    saver = tf.train.Saver(max_to_keep=5)#保存最近的几个模型
    # start a tf session with certain config gpu训练选项 meishayong
    #config = tf.ConfigProto()    
    #config.gpu_options.allow_growth = True    
    #with tf.Session(config = config) as sess:
    #with tf.Session() as sess,tf.device('/gpu:0'):
    with tf.Session() as sess:   
        
        #saver.restore(sess, "output/saver-1000")
        print("Session started......")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()#创建Coord类来协同多线程
        threads = tf.train.start_queue_runners(coord = coord)
        step = 0
        #summary_writer = tf.summary.FileWriter('output', sess.graph)
        while (True):
            step += 1;
            try:

                # get real data from tfrecord first
                z_, x_, z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_= sess.run([z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h])
                
                # calculate crop size for z,x
                context_z = 0.5*(z_target_w_+z_target_h_)#p=0.25(w+h)
                z_sz = tf.cast(tf.sqrt(tf.constant(z_target_w_+context_z)*tf.constant(z_target_h_+context_z)), tf.float64)#(w +2p)*(h+2p)

                context_x = 0.5*(x_target_w_+x_target_h_)
                x_sz = tf.cast(tf.sqrt(tf.constant(x_target_w_+context_x)*tf.constant(x_target_h_+context_x)), tf.float64)#(w +2p)*(h+2p)
                x_sz = float(255) / 127 * x_sz#xsz是zsz的255/127倍
                z_sz_, x_sz_ = sess.run([z_sz, x_sz])

                # input z into conv net to get its feature map        
                templates_z_, z_crops_ = sess.run([templates_z, z_crops], feed_dict={
                    siamNet.batched_pos_x_ph: z_pos_x_,
                    siamNet.batched_pos_y_ph: z_pos_y_,
                    siamNet.batched_z_sz_ph: z_sz_,
                    image: z_})
                # visualize croped z image
#                """             
#                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#                cv2.circle(z_crops_[0],(int(design.exemplar_sz / 2.),int(design.exemplar_sz / 2.)),2,(255,125,125),-11)
#                label_w = int(z_target_w_[0] * design.exemplar_sz / z_sz_[0])
#                label_h = int(z_target_h_[0] * design.exemplar_sz / z_sz_[0])
#                cv2.rectangle(z_crops_[0], (int(design.exemplar_sz / 2. - label_w / 2), int(design.exemplar_sz / 2. - label_h / 2)), (int(design.exemplar_sz / 2. + label_w / 2), int(design.exemplar_sz / 2. + label_h / 2)), (255,0,0), 2)
#      
#                cv2.imshow('image',z_crops_[0])
#                cv2.waitKey(0)
#                return
#                """
                
                # create ground truth response map根据z图像大小255和gt创建label，以原点为中心xw、xh的矩形框标签为1，剩下为0
                label = _create_gt_label_final_score_sz(8, final_score_sz, x_target_w_, x_target_h_, x_sz_, 255)
                
 
#final_score_sz：255
#batched_data
#image：=shape=(8, ?, ?, 3)image原图像大小700, 
#templates_z：[24, 17, 17, 32], scores：#[24, 255, 255, 1], 
#loss = tf.reduce_mean(tf.log(1 + tf.exp(-score * self.label))), 
#train_step= tf.train.AdamOptimizer(1e-6).minimize(loss), 
#distance_to_gt是一个数 为了可视化训练过程中的距离值而已，没有真实含义, 
#summary= tf.summary.scalar('distance_to_gt', distance_to_gt)#可视化训练过程
               
                # input x into net, get x feature map, calculate score map, and its loss, iterate one train step
                scores_, loss_, _, x_crops_, summary_, distance_to_gt_,templates_x_,net_cat_= sess.run(
                    [scores, loss,  train_step, x_crops, summary, distance_to_gt, templates_x, net_cat],
                    feed_dict={                       
                    siamNet.batched_z_sz_ph: z_sz_,
                    siamNet.batched_pos_x_ph: x_pos_x_,
                    siamNet.batched_pos_y_ph: x_pos_y_,
                    siamNet.batched_x_sz0_ph: x_sz_,
                    siamNet.batched_x_sz1_ph: x_sz_ * 0.98,
                    siamNet.batched_x_sz2_ph: x_sz_ * 1.02,
                    templates_z: np.squeeze(templates_z_),
                    image: x_,
                    siamNet.label: label
                    })#两个feeddict过程可以合成一个？？
                
                
#                # visualize the output score map
#                """
##                if step % 2 == 0:
##                    plt.subplot(121)
##                    plt.imshow(np.squeeze(scores_[0] + 10)/20 , cmap = 'gray')
##                    plt.subplot(122)
##                    plt.imshow(x_crops_[0] + 0.5)
##                    plt.show()
##                    plt.pause(5)
##                    """
##
#
#                # visualize croped x image
#                """
#                cv2.namedWindow('image', cv2.WINDOW_NORMAL)              
#                label_w = int(x_target_w_[0] * design.search_sz / x_sz_[0])
#                label_h = int(x_target_h_[0] * design.search_sz / x_sz_[0])
#                cv2.rectangle(x_crops_[0], (int(- label_w / 2 + 257 / 2), int( - label_h / 2+ 257 / 2)), (int( + label_w / 2+ 257/2), int(+ label_h / 2+ 257/2)), (255,0,0), 2)
#                cv2.imshow('image',x_crops_[0] + 0.5)                
#                cv2.waitKey(0)
#                """
                
                if step % 5 == 0:
                    print("step %d, loss=%f, distance_to_gt=%f"%(step, loss_, distance_to_gt_))

                    if step % 500 == 0:
                        saver.save(sess, os.path.join("output_vgg", "saver-test") , global_step = step)
                #main(step)

            except tf.errors.OutOfRangeError:
                print("End of training")  # ==> "End of dataset"
                break
        
        # Finish off the fiqlename queue coordinator.
        coord.request_stop()
        coord.join(threads)   

          
def _create_gt_label_final_score_sz(batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz, search_sz):
    label = [[[-1. for y_coor in range(final_score_sz)] for x_coor in range(final_score_sz)] for c in range(batch_size)]
    for i in range(batch_size):
        label_w = int(x_target_w_[i] * search_sz / x_sz[i])
        label_h = int(x_target_h_[i] * search_sz / x_sz[i])
        for x_index in range(label_w):
            for y_index in range(label_h):
                label[i][int(final_score_sz / 2. + y_index - label_h / 2.)][int(final_score_sz / 2. + x_index - label_w / 2.)] = 1.

    return label
