import tensorflow as tf
from absl.flags import FLAGS

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    
    #这个函数分别对比某一类anchors (一共是三类，每一类对应不同的尺寸的box)
    #每一类box 对应的尺寸翻倍
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    #这里的Ｎ是样本的数量
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    
    #输出的张量尺寸
    
    #tf.shape(anchor_idxs)=3=len(anchor_idxs)
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    
    #这是动态数组
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    
    #N 对应的是样本数量
    #二维遍历，i对应的是每一个样本
    for i in tf.range(N):
        
        
        #tf.shape(y_true)=[  N, 100,   6]
        #一 张图片最多识别100个目标，因为一幅图最多对应100个
        for j in tf.range(tf.shape(y_true)[1]):
            
            
            """
            ++++++++++++++++ x2,y2
            +                +
            +                +
            +                +
            x1,y1 ++++++++++++
            """
            
            # x2=y_true[i][j][2]对应的是标记的矩形终点坐标
            #如果x2==0那么就没有这个类别　pass
            if tf.equal(y_true[i][j][2], 0):
                continue
            
            #这里指的是y_true[i][j][5] 这个种类的anchor 是否在这个　anchor_idxs中
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
            
            print(anchor_eq)
            
            # print(anchor_idxs.numpy(), '##############',y_true[i][j][5].numpy())
            
            print('-'*30+'>')
            # print(i,j)
            #如果y_true[i][j][5] 这个种类的anchor 是在这个　anchor_idxs中
            #即　anchor_idxs 存在一个　值为True 
            if tf.reduce_any(anchor_eq):
                
                
                #这是box的坐标
                box = y_true[i][j][0:4]
                
                #box 的中点坐标
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2


                #找到标注的那个box 对应的anchor 对应的位置，这里重新编码了
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)

                
                
                #grid_xy是grid_size*grid_size　这个真实　box下anchor中心的坐标
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    #id   i=样本编号(0-6)，anchor中心坐标x,y    anchor 种类取值在[0,1,2]
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    
                    #对应的标注坐标　和　　　　　　　　#1只是占位　　类别
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())
    
    #y_true_out.shape=[3, 104, 104, 3, 6]
    #3是样本数量
    #104是指的是box 的大小,每一个pixel都有可能是anchor 的中心点
    #所以就粗暴的给每一个pixel分配了一个内存空间
    #3 是同一个尺度的anhor 点有３个box 
    #6对应　[x1, y1, x2, y2, class , anchor_class]
    #
    #
    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    
    y_outs = []
    #将图像分成32*32格
    
    #grid_size=13
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    
    #anchors 是聚类出来的点,x,y分别是聚类框框的宽度和高度
    
    #这里是每个anchor 框框的面积
    anchor_area = anchors[..., 0] * anchors[..., 1]
    
    #box_wh的宽度-高度,　box_wh.shape＝[k, 100, 2],k是样本的数量
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    
    #这里将box_wh从三维扩张到四维
    box_wh_expand=tf.expand_dims(box_wh, -2)
    # tf.tile是将量在某个或某几个维度上复制,这里是在第三个维度上复制,复制９个，因为一共９个锚点
    
    #box_wh.shape=[3, 100, 9, 2],从原来的一行两列变成９行两列
    
    '''
    box_wh[0][0]=
    
    <tf.Tensor: shape=(9, 2), dtype=float32, numpy=
array([[0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998],
       [0.55466664, 0.32999998]], dtype=float32)>
    
    '''
    
    box_wh = tf.tile(box_wh_expand,
                     (1, 1, tf.shape(anchors)[0], 1))
    
    
    #box_area.shape=[k, 100, 9]
    '''
    
    box_area[0][0]
    Out[362]: 
    <tf.Tensor: shape=(9,), dtype=float32, numpy=
    array([0.18303998, 0.18303998, 0.18303998, 0.18303998, 0.18303998,
           0.18303998, 0.18303998, 0.18303998, 0.18303998], dtype=float32)>
    '''
    
    box_area = box_wh[..., 0] * box_wh[..., 1]
    
    
    #tf.minimum(A,B),　Ａ的维度为mn,B的维度为kn,且m=n,或者　n=1,就可以比较大小
    #intersection是交集
    #这里用到了矩阵的广播机制，分别与９个anchor box 进行比较
    #                           delta  x
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])  #delta y
    
    #交并比
    iou = intersection / (box_area + anchor_area - intersection)
    
    
    #找到和标记的框框最接近那个anchor ,输出anchor_id 
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    #这里的y_train.shape=3, 100, 6],最后一个维度是６
    #[x1,y1,x2,y2,class_id,anchor_idx]
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    
    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


import pickle
import numpy as np

# data_output = open('data.pkl','wb')
# pickle.dump(kk1,data_output)
# data_output.close()


# rb 以二进制读取
data_input = open('data.pkl','rb')
y_train = pickle.load(data_input)
data_input.close()

size=416




anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

 #y_train.shape=(6, 100, 5)
 #6是样本数量
 #100是标签数量
 # 5[x1,y1,x2,y2,class]

cc= transform_targets(y_train, anchors, anchor_masks, size)