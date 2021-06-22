#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:36:13 2021

@author: ledi
"""


import pickle
import numpy as np


from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
# from utils1 import broadcast_iou

# flags.DEFINE_integer('yolo_max_boxes', 100,
#                      'maximum number of boxes per image')
# flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
# flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
#                               (81, 82), (135, 169),  (344, 319)],
#                              np.float32) / 416
# yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])




# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    
    # pred=output_0
    # anchors=anchors[masks[0]]
    
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    
    print('grid_size=',grid_size)
    
    #将85 维度的向量分割成　2+2+1+classes
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)


    #进行sigmoid 变换
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    
    #没有归一化的box_wh与sigmoid 之后的box_xy
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]


    #https://www.cnblogs.com/wangxinzhe/p/10648465.html
    #https://www.shuzhiduo.com/A/qVdeERkndP/
    
    
    
    
    box_xy = (tf.cast(box_xy,tf.float32) + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FLAGS.yolo_max_boxes= 100
FLAGS.yolo_iou_threshold 0.5
FLAGS.yolo_score_threshold 0.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    
    
    #     min_x2       box_1的x2    box_2的x2            box_1的x1            box_2的x1
    #最小值减去最大值
    #例如    b1x1 --b2x1 ----  b1x2---b2x2
    #  X轴---------------------------->
    #min_w=b1x2-b2x1
    min_w=tf.minimum(box_1[..., 2], box_2[..., 2]) -tf.maximum(box_1[..., 0], box_2[..., 0])
    int_w = tf.maximum(min_w, 0)
    #     min_y2       box_1的y2    box_y的x2            box_1的y1            box_2的y1
    min_h=tf.minimum(box_1[..., 3], box_2[..., 3]) -tf.maximum(box_1[..., 1], box_2[..., 1])
    int_h = tf.maximum(min_h, 0)
    
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        # 真实标注的矩形框　-  置信度，表示这个box 是否有目标 -类别id 　      
        true_box, 　　　　　　　　true_obj, 　　　　　　　　　true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        
        '''
        kkk=true_xy[0]
        for a0 in range(kkk.shape[0]):
            for a1 in range(kkk.shape[1]):
                for a2 in range(kkk.shape[2]):
                    for a3 in range(kkk.shape[3]):
                        if  0<abs(kkk[a0][a1][a2][a3])<1:
                            print(kkk[a0][a1][a2][a3])
                            print([a0,a1,a2,a3])

        '''
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        
        
        #这里并不学习wh,而是学习exp(wh)
        true_wh = tf.math.log(true_wh / anchors)
        
                       
        #tf.math.is_inf(true_wh) 判断数值是否是　inf
        
        #tf.zeros_like(true_wh).shape==true_wh.shape,但是元素的值全是0
        #https://blog.csdn.net/luoganttcc/article/details/118091981
        '''
        tf.where(
               condition, x=None, y=None, name=None)
        
        [if  condition[k] =True  取　x[k] else 取　y[k]]
        '''
        
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        #obj_mask.shape=TensorShape([8, 13, 13, 3])
        #obj_mask　用来计算loss 会将没有标注的那些anchor loss置为0 ,只考虑有标记的box 
        #https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py　第36行
        obj_mask = tf.squeeze(true_obj, -1)
        
        '''
        kkk=obj_mask[0]
        for a0 in range(kkk.shape[0]):
            for a1 in range(kkk.shape[1]):
                for a2 in range(kkk.shape[2]):
                    # for a3 in range(kkk.shape[3]):
                        if  kkk[a0][a1][a2]==1:
                            print(kkk[a0][a1][a2])
                            print([a0,a1,a2])
                            
        '''
        
        # ignore false positive when iou is over threshold
        
        '''
        box_2=tf.boolean_mask(
               true_box, tf.cast(obj_mask, tf.bool))
        box_1=pred_box
        '''
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        '''
        #对于broadcast_iou　可以等价与下面的一串代码，
        #但是broadcast_iou中用了张量的广播机制，
        iou=[]
        for i in range(8):
            for j in range(13):
                for k in range(13):
                    for n in range(3):
                        if true_box[i,j,k,n][2]>0:
                            # print(true_box[i,j,k,n])
                            # print(pred_box[i,j,k,n])
                            print([i,j,k,n])
                            
                            t=true_box[i,j,k,n]
                            p=pred_box[i,j,k,n]
                            min_w=tf.minimum(p[ 2], t[ 2]) -tf.maximum(p[ 0], t[0])
                            int_w = tf.maximum(min_w, 0)
                            
                            min_h=tf.minimum(p[ 3], t[ 3]) -tf.maximum(p[ 1], t[ 1])
                            int_h = tf.maximum(min_h, 0)
                            int_area = int_w * int_h
                            
                            p_area = (p[2] - p[ 0]) * (p[3] - p[1])
                            t_area = (t[ 2] - t[ 0]) * (t[ 3] - t[ 1])
                            
                            temp_iou=int_area / (p_area + t_area - int_area)
                            iou.append([[i,j,k,n],temp_iou])
                            print('------------------------->',temp_iou)
                            # print('-----------------------------')
        '''

        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        
        #终点的loss= obj_mask * box_loss_scale*sum((px1-tx1)**2+(px2-tx2)**2)
        #因为这里有obj_mask的存在，只考虑有目标点的loss 
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            
        #置信度损失,表示这个box 是否有物体
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        
        #obj_mask * obj_loss 
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)


        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss




# model_input = open('train.pkl','wb')


# # pred=model.predict(k[0])



# for i in [0,1,2]:
#     y_pred = open('pred'+str(i)+'.pkl','wb')
#     pickle.dump(pred[i],y_pred)
#     y_pred.close()
    
    
# for j in [0,1,2]:
#     true = open('true'+str(j)+'.pkl','wb')
#     pickle.dump(k[1][j],true)
#     true.close()



y_truek=[]

for j in [0,1,2]:
    true = open('true'+str(j)+'.pkl','rb')
    true_array=pickle.load(true)
    y_truek.append(   tf.constant(true_array))
    true.close()

y_truek=tuple(y_truek)



y_predk=[]

for j in [0,1,2]:
    pred1 = open('pred'+str(j)+'.pkl','rb')
    pred_array=pickle.load(pred1)
    
    y_predk.append(tf.constant(pred_array))
    pred1.close()

y_predk=tuple(y_predk)

# loss = [YoloLoss(yolo_anchors[mask], classes=20)
#         for mask in yolo_anchor_masks]


mask=yolo_anchor_masks[0]

anchors=yolo_anchors[mask]
ignore_thresh=0.5


loss0=YoloLoss(yolo_anchors[mask], classes=20)(y_truek[0],y_predk[0])



y_true=  y_truek[0]

y_pred=  y_predk[0]

classes=20
# Lg08300734314159$


# # rb 以二进制读取
# y_true1 = open('y_true.pkl','rb')
# y_train = pickle.load(y_true1)
# # data_input.close()

# # size=416
