# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:37:59 2018

@author: emily
"""
import tensorflow as tf
from crop import pad_frame
from crop import extract_crops_z
from crop import extract_crops_x
#from convolutional import create_siamese_train #alexnet结构
from vgg19 import create_siamese_train #vgg19结合注意力机制结构
_bnorm_adjust = True
class Siamese(object):
    def __init__(self, batch_size):
	    # define all the placeholders in the net
        self.batch_size =batch_size
        self.batched_pos_x_ph = tf.placeholder(tf.float64, shape = [self.batch_size])#居中后的x,y的真实值，原点在图片左上角
        self.batched_pos_y_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_z_sz_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_sz0_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_sz1_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_sz2_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_pos_x = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_pos_y = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_target_w = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.batched_x_target_h = tf.placeholder(tf.float64, shape = [self.batch_size])
        self.label = tf.placeholder(tf.float32, [self.batch_size, None, None])
    
    def build_tracking_graph_train(self, final_score_sz = 255):
        



        image = tf.placeholder(tf.float32, [self.batch_size] + [None, None, 3], name = "input_image")
#        shape=(8, ?, ?, 3)image原图像大小700
        frame_sz = tf.shape(image)[1:]#shape(image)除去batch_size维，一共有3维,输出为：？？3	
#		# used to pad the crops
        avg_chan = tf.reduce_mean(image, axis=(1, 2), name='avg_chan')#2维(8,3),同时沿着第1,2维求平均，也就是整张图求像素平均
#        print(avg_chan[1].get_shape())

#		 pad with if necessary
        single_crops_z = []
        single_crops_x = []
        
		#裁剪一个batch里面的每一个图像slice a batch into single images, and crop them one by one
        for batch in range(self.batch_size):
            single_pos_x_ph = self.batched_pos_x_ph[batch]#模板图像z的x
            single_pos_y_ph = self.batched_pos_y_ph[batch]#模板图像z的y
            single_z_sz_ph = self.batched_z_sz_ph[batch]#模板图像z的size#(w +2p)*(h+2p)！！！
            single_x_sz0_ph = self.batched_x_sz0_ph[batch]#搜索图像x的size0=zsz*255/127
            single_x_sz1_ph = self.batched_x_sz1_ph[batch]#搜索图像x的size1=zsz*255/127
            single_x_sz2_ph = self.batched_x_sz2_ph[batch]#搜索图像x的size2=zsz*255/127*1.02
			
			#pad crop z
#frame_padded_z, npad_z = pad_frame(single_z=batch内某一张图片size[700,700,3], frame_sz = [?,?,3], 
#single_pos_x_ph, single_pos_y_ph模板图片的真实xy, single_z_sz_ph是(w +2p)*(h+2p), avg_chan[batch1]的size(none,none,3))
            single_z = image[batch]
            frame_padded_z, npad_z = pad_frame(single_z, frame_sz, single_pos_x_ph, single_pos_y_ph, single_z_sz_ph, avg_chan[batch])
            frame_padded_z = tf.cast(frame_padded_z, tf.float32)#输出填充好的一张图片张量size[填充值2*npad+zsize，填充值2*npad+zsize，3]
#			# extract tensor of z_crops#squeeze,默认删除1的维度，也可指定维度进行删除
            single_crops_z.append(tf.squeeze(extract_crops_z(frame_padded_z, npad_z, single_pos_x_ph, single_pos_y_ph, single_z_sz_ph, 127)))
			
			# pad crop x
            single_x = image[batch]#和上面相比，single_x_sz2_ph是single_z_sz_ph的255/127*1.02倍
            frame_padded_x, npad_x = pad_frame(single_x, frame_sz, single_pos_x_ph, single_pos_y_ph, single_x_sz2_ph, avg_chan[batch])
            frame_padded_x = tf.cast(frame_padded_x, tf.float32)
			
#			# extract tensor of x_crops (3 scales)
            single_crops_x.append(tf.squeeze(extract_crops_x(frame_padded_x, npad_x, single_pos_x_ph, single_pos_y_ph, single_x_sz0_ph, single_x_sz1_ph, single_x_sz2_ph, 255)))
#
		# stack the cropped single images
        z_crops = tf.stack(single_crops_z)#(8, 127, 127, 3)
        x_crops = tf.stack(single_crops_x)#(8, 3, 255, 255, 3)
		
        x_crops_shape = x_crops.get_shape().as_list()#[8, 3, 255, 255, 3]
        x_crops = tf.reshape(x_crops, [x_crops_shape[0] * x_crops_shape[1]] + x_crops_shape[2: ])		
        print("shape of single_crops_x: ", single_crops_x[0].shape, "shape of x_crops: ", x_crops.shape)
        print("shape of single_crops_z: ", single_crops_z[0].shape, "shape of z_crops: ", z_crops.shape)
#		
        
        
        
        ##############################
#		# use crops as input of  fully-convolutional Siamese net
        #alex
#        template_z, templates_x = create_siamese_train(x_crops, z_crops, h=[11, 5, 3, 3, 3],w=[11, 5, 3, 3, 3],num=[96, 128, 96, 96, 32])
        #vgg19
        #net_z, net_x=create_siamese_train(batch1,batch2,17,49)
        #net = vgg19.Vgg19()
        #x_featuresize = 17
        #batchsize_z = 8
        #batchsize_x = 8
        #template_z = net.build_network(x_crops,x_featuresize,batchsize_z,False)
        #templates_x = net.build_network(z_crops,z_featuresize,batchsize_x,True)
        template_z,templates_x = create_siamese_train(z_crops,x_crops,17,49,8,24)#第一次反向传播以后，产生的图像特征向量都为nan
        print("shape of template_z:", template_z.shape)#(8,17,17,32)
##		
        
        
        
#		# extend template_z to match the triple scaled feature map of x
        template_z_list = []
        for batch in range(self.batch_size):
            template_z_list.append(template_z[batch])
            template_z_list.append(template_z[batch])
            template_z_list.append(template_z[batch])
            templates_z = tf.stack(template_z_list)#32->1472
        print("shape of templates_z:", templates_z.get_shape().as_list())#[24, 17, 17, 32][24, 17, 17, 1472]
        print("shape of templates_x:", templates_x.get_shape().as_list())#[24, 49, 49, 32][24, 49, 49, 1472]
#		
#		# compare templates via cross-correlation互相关层 
        scores,net_cat = self._match_templates_train(templates_z, templates_x)#netcat，templatez,x没有值
#        # resize to final_score_sz双线性插值放大到255
        scores_up = tf.image.resize_bilinear(scores, [final_score_sz, final_score_sz], align_corners=True)
        print("shape of big score map:", scores_up.get_shape().as_list())#[24, 255, 255, 1]#有值
#		
#		# only choose one scale for each image每张图片选择一个规模大小计算分数图 这里可以选
        score = tf.squeeze(tf.stack([scores_up[i]  for i in [0 + 3 * i for i in range(self.batch_size)]]))
        print("shape of score map:", score.get_shape().as_list())#[8, 255, 255]
        
        loss = self.cal_loss(score)
        
       
        distance_to_gt, max_pos_x, max_pos_y = self.distance(score, final_score_sz)
        train_step = tf.train.AdamOptimizer(10e-9).minimize(loss)
        summary = tf.summary.scalar('distance_to_gt', distance_to_gt)#可视化训练过程
#		
        return image, z_crops, x_crops, templates_z, scores_up, loss, train_step, distance_to_gt, summary,templates_x,net_cat
 




    
#scores = self._match_templates_train(templates_z=size[24, 17, 17, 32], templates_x=size[24, 49, 49, 32])    
    def _match_templates_train(self, net_z, net_x):#互相关层
		# finalize network
		# z, x are [B, H, W, C]
        print("shape_net_z:", net_z.shape)#(24, 17, 17, 1472
        net_z = tf.transpose(net_z, perm=[1,2,0,3])
        net_x = tf.transpose(net_x, perm=[1,2,0,3])
		# z, x are [H, W, B, C]获得他们分别是多少，以便reshape一个batch所有图像
        Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
		# assert B==Bx, ('Z and X should have same Batch size')
		# assert C==Cx, ('Z and X should have same Channels number')
        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, Bx*Cx))
        #第一个参数input：指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
        #第二个参数filter：相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, in_channels, channel_multiplier]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，输入通道数，输出卷积乘子]，同理这里第三维in_channels，就是参数value的第四维
        #第三个参数strides：卷积的滑动步长。
        #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同边缘填充方式。
        net = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
		    #结果返回一个Tensor，shape为[batch, out_height, out_width, in_channels * channel_multiplier（filter的第四维=1）]
        print("shape of net:", net.get_shape().as_list())#[1, 33, 33, 35328]
		# final is [1, Hf, Wf, BC]B24 C32
        net_cat = tf.concat(tf.split(net, 3 * self.batch_size, axis=3), axis=0)
        print("shape of net_cat:", net_cat.get_shape().as_list())# [24, 33, 33, 1472]
		# final is [B, Hf, Wf, C]
		#
        net_final = tf.reduce_mean(net_cat, axis=3, keep_dims = True)
		#net_final = tf.Print(net_final, [net_final], summarize = 100)
		#net_final = tf.Print(net_final, [net_final])
		

		# final is [B, Hf, Wf, 1]

	
        if _bnorm_adjust:#在互相关层做了batchnorm
            net_final = tf.layers.batch_normalization(net_final)
            print("shape of net_final:", net_final.get_shape().as_list())
		
        return net_final,net# final is [B, Hf, Wf, 1]
    
#distance_to_gt, max_pos_x, max_pos_y = self.distance(score=size[8, 255, 255], final_score_sz=255)    
    def distance(self, score, final_score_sz):
        if (self.batch_size == 1):
            score = tf.reshape(score, [1] + score.get_shape().as_list())
		# reshape to flatten the score map
        flat_scores = tf.reshape(score, [self.batch_size, score.get_shape().as_list()[1] * score.get_shape().as_list()[2]])
		# find the index of the maximum score on the flat score map
#        这个函数的作用是返回 input 中每行（最后一维）最大的 k 个数，并且返回它们所在位置的索引
        _, max_pos = tf.nn.top_k(flat_scores, 3)	#(8,3)	
        max_pos = tf.reduce_mean(max_pos, axis = 1)#(8)
		
		# convert the index to 2D
        max_pos_y = max_pos // final_score_sz#除后取整[8]
        max_pos_x = max_pos % final_score_sz#除后取余[8]
        print("shape of max response distance_to_gt:",max_pos_x.get_shape().as_list())
        #distance_to_gt是一个数 为了可视化训练过程中平均最热点和中心点距离值而已，没有真实含义
        distance_to_gt = tf.reduce_mean(tf.sqrt(tf.square(final_score_sz / 2. - tf.cast(max_pos_x, tf.float32)) + tf.square(final_score_sz / 2. - tf.cast(max_pos_y, tf.float32))))
#        print("shape of max response distance_to_gt:",distance_to_gt.get_shape().as_list(),np.shape(distance_to_gt))
        return distance_to_gt, max_pos_x, max_pos_y


    def cal_loss(self, score):
		#calculate logistic loss of score map   
        loss = tf.reduce_mean(tf.log(1 + tf.exp(-score * self.label)))		
        return loss    