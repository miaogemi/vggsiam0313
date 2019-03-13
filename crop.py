from __future__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
import functools
#frame_padded_z, npad_z = pad_frame(single_z=batch内某一张图片size[none,none,3], frame_sz = [?,?,3], 
#single_pos_x_ph, single_pos_y_ph700大小模板图片的xy, single_z_sz_ph是根号下(w +2p)*(h+2p)=a=2c, avg_chan[batch1]的size(none,none,3))
def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    #保证各个边距离中心点都是c，以后才方便裁剪，输出被填充好的一张图片向量
    c = patch_sz / 2
    xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
    ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
    xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
    ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])
    npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])#取最大需要填充值作为npad
    paddings = [[npad, npad], [npad, npad], [0, 0]]
    im_padded = im
    if avg_chan is not None:
        #print("shape of im and avg_chan: ", im.shape, avg_chan.shape)
        im_padded = im_padded - avg_chan#图像像素减去平均像素
    #print("shape of pad image: ", im_padded.shape)
    im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')#把0填充进宽为npad的边界
    if avg_chan is not None:
        im_padded = im_padded + avg_chan#统一加上平均像素值
    return im_padded, npad#输出tensor的size[填充值2*npad+imsize700，填充值2*npad+imsize700，3]

#extract_crops_z(frame_padded_z, npad_z上个函数输出的impaded和npad, 
#single_pos_x_ph, single_pos_y_ph是z图像的xy, single_z_sz_ph同上, design.exemplar_sz = 127)
def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
    width = tf.round(pos_x + c) - tf.round(pos_x - c)
    height = tf.round(pos_y + c) - tf.round(pos_y - c)
    crop = tf.image.crop_to_bounding_box(im,#解释参考代码说明
                                         tf.cast(tr_y, tf.int32),#图像左上点的yoffset_height：结果图像左上角点的垂直坐标
                                         tf.cast(tr_x, tf.int32),#offset_width：结果图像左上角点的水平坐标 
                                         tf.cast(height, tf.int32),#target_height：结果图像的高度 
                                         tf.cast(width, tf.int32))#target_width：结果图像的宽度
    crop = tf.image.resize_images(crop, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)#缩放到127。127。3  
    crops = tf.expand_dims(crop, axis=0)#在0维增加维度
    return crops#[1,127,127,3]

#extract_crops_x(frame_padded_x, npad_x, 
#single_pos_x_ph, single_pos_y_ph,是x图像的xy，
# single_x_sz0_ph=zsz*255/127, single_x_sz1_ph=zsz*255/127, single_x_sz2_ph=zsz*255/127*1.02, 255）
def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # take center of the biggest scaled source patch
    c = sz_src2 / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
    tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = tf.round(pos_x + c) - tf.round(pos_x - c)
    height = tf.round(pos_y + c) - tf.round(pos_y - c)
    search_area = tf.image.crop_to_bounding_box(im,
                                                tf.cast(tr_y, tf.int32),
                                                tf.cast(tr_x, tf.int32),
                                                tf.cast(height, tf.int32),
                                                tf.cast(width, tf.int32))
    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2

    crop_s0 = tf.image.crop_to_bounding_box(search_area,
                                            tf.cast(offset_s0, tf.int32),
                                            tf.cast(offset_s0, tf.int32),
                                            tf.cast(tf.round(sz_src0), tf.int32),
                                            tf.cast(tf.round(sz_src0), tf.int32))
    crop_s0 = tf.image.resize_images(crop_s0, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crop_s1 = tf.image.crop_to_bounding_box(search_area,
                                            tf.cast(offset_s1, tf.int32),
                                            tf.cast(offset_s1, tf.int32),
                                            tf.cast(tf.round(sz_src1), tf.int32),
                                            tf.cast(tf.round(sz_src1), tf.int32))
    crop_s1 = tf.image.resize_images(crop_s1, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crop_s2 = tf.image.resize_images(search_area, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crops = tf.stack([crop_s0, crop_s1, crop_s2])
    return crops#[3,255,255,3]