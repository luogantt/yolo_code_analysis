#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 10:24:03 2021

@author: ledi
"""

import pickle
import numpy as np
import tensorflow as tf

# data_output = open('output_0.pkl','wb')
# pickle.dump(output_0,data_output)
# data_output.close()



'''

if n_a= n_b=13

aa=bb=Out[297]: 
<tf.Tensor: shape=(13, 13), dtype=int32, numpy=
array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
       [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
       [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
       [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
       [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
       [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
       [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
       [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]], dtype=int32)>
'''


def _meshgrid(n_a, n_b):
    
    aa=tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a))
    
    bb=tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    

    return [aa,bb]  


def yolo_boxes(pred, anchors, classes):
    
    # pred=output_0
    # anchors=anchors[masks[0]]
    
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    
    #获取特征的维度
    #在yolov3 中有下面三种维度
    
    '''
    grid_size= tf.Tensor([13 13], shape=(2,), dtype=int32)
    grid_size= tf.Tensor([26 26], shape=(2,), dtype=int32)
    grid_size= tf.Tensor([52 52], shape=(2,), dtype=int32)
    '''
    grid_size = tf.shape(pred)[1:3]   #13*13
    
    

    
    #将85 维度的向量分割成　2+2+1+classes
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)
    
    #中点坐标
    box_xy = tf.sigmoid(box_xy)
    #置信度
    objectness = tf.sigmoid(objectness)
    #softmax分类
    class_probs = tf.sigmoid(class_probs)
    
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]


    #https://www.cnblogs.com/wangxinzhe/p/10648465.html
    #https://www.shuzhiduo.com/A/qVdeERkndP/
    
    
    '''
    tf.cast(box_xy,tf.float32).shape
    TensorShape([5, 13, 13, 3, 2])
    
    tf.cast(grid, tf.float32).shape
    TensorShape([13, 13, 1, 2])
    '''
    # vv=tf.cast(box_xy,tf.float32).shape=TensorShape([5, 13, 13, 3, 2])
    # cc=tf.cast(grid, tf.float32).shape=    TensorShape([13, 13, 1, 2])
    
    #k=vv[0]+cc
    
    #box_xy_shift[0]==k
    
    #每一个grid 负责一次检测
    ##grid 为偏移 ，将x,y相对于featuremap尺寸进行了归一化 
    
    box_xy_shift= (tf.cast(box_xy,tf.float32) + tf.cast(grid, tf.float32))
    
    box_xy = box_xy_shift / tf.cast(grid_size, tf.float32)
    
    box_wh = tf.exp(box_wh) * anchors/ tf.cast(grid_size, tf.float32)

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box



anchors0 = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


anchors=anchors0[masks[0]]



# rb 以二进制读取
data_input = open('output_0.pkl','rb')
pred= pickle.load(data_input)
data_input.close()

classes=80

pm=pred.numpy()

pred=pred/pm.max()
cc=yolo_boxes(pred, anchors, classes)


print(cc)

