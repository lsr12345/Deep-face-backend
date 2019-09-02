# coding: utf-8

import io
from flask import Flask, request, send_file
from object_detection.utils import ops as utils_ops
import os
import numpy as np
import cv2
from gevent import monkey
import tensorflow as tf
import datetime

# In[ ]:

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
monkey.patch_all()
app = Flask(__name__)

PATH_TO_FROZEN_GRAPH = "./models/frozen_inference_graph_detection.pb"
PATH_TO_LABELS = "object_detection/face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False

detection_sess = tf.Session(config=config)

with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        
#############################################################
###face_feature
face_feature_sess = tf.Session(config=config)
ff_pb_path = './models/frozen_inference_graph_recognition.pb'
with face_feature_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        ff_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(ff_od_graph_def, name='')
        ff_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        ff_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        ff_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

##############################################################
### face_lanmark
face_landmark_sess = tf.Session(config=config)
fl_pb_path = './models/landmark.pb'
with face_landmark_sess.as_default():
    fl_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(fl_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        fl_images_placeholder = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        fl_logits = tf.get_default_graph().get_tensor_by_name("fully_connected_9/Relu:0")

###################################################################
# face_change men2women
face_change_sess = tf.Session()
fc_pb_path = './models/men2women.pb'
with face_change_sess.as_default():
    fc_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(fc_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        fc_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(fc_od_graph_def, name='')
        
        fc_images_placeholder = tf.get_default_graph(),get_tensor_by_name('input_image:0')
        fc_output = tf.get_default_graph().get_tensor_by_name('output_image:0')
        
###################################################################
# face_attribute men2women
face_attribute_sess = tf.Session()
fa_pb_path = './models/face_attribute.pb'
with face_attribute_sess.as_default():
    fa_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(fc_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        fa_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(fa_od_graph_def, name='')

        fa_images_placeholder = tf.get_default_graph(),get_tensor_by_name('Placeholder:0')
        predict_eyeglasses = tf.get_default_graph().get_tensor_by_name('Softmax:0')
        predict_young = tf.get_default_graph().get_tensor_by_name('Softmax_1:0')
        predict_male = tf.get_default_graph().get_tensor_by_name('Softmax_2:0')
        predict_smiling = tf.get_default_graph().get_tensor_by_name('Softmax_3:0')        
        
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    return upload_path

@app.route("/face_detect")
def inference():
    im_url = request.args.get("url")

    im_data = cv2.imread(im_url)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            y1 =   bbox[0]
            x1 =  bbox[1]
            y2 =   (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

    return str([x1, y1, x2, y2])

# 图像数据标准化处理
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def read_image(path):
    im_data = cv2.imread(path)
    im_data = prewhiten(im_data)
    im_data = cv2.resize(im_data, (160, 160))
    # 1 *  H * W * 3  
    return im_data
    
@app.route('/face_feature')
def face_feature():
    im_data1 = read_image('')
    im_data1 = np.expand_dims(im_data1, axis=0)
    emb_1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})
    strr = ','.join(str(i) for i in emb_1[0])
    return strr

@app.route('/face_register', methods=['POST', 'GET'])
def face_register():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    upload_path = os.path.join("tmp/tmp_register" + time + '.' + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 只返回检测得分最高的人脸
    l = np.array(output_dict['detection_scores']).argsort()
    i = l[-1]
    mess = 'fail'
    if output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        y1 =   bbox[0]
        x1 =  bbox[1]
        y2 =   (bbox[2])
        x2 = (bbox[3])
        print(output_dict['detection_scores'][i], x1, y1, x2, y2)
        # 利用obdetection检测出来物体的坐标为相对于图片归一化后的值，0~1
        y1 = int(y1*sp[0])
        x1 = int(x1*sp[1])
        y2 = int(y2*sp[0])
        x2 = int(x2*sp[1])
        im_data = im_data[y1:y2, x1:x2]
        im_data = prewhiten(im_data)
        im_data = cv2.resize(im_data, (160, 160))
        im_data = np.expand_dims(im_data, axis=0)
        emb_1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data, ff_train_placeholder:False})
        strr = ','.join(str(i) for i in emb_1[0])
        # 人脸特征写入txt中
        with open('face/feature.txt', 'w') as f:
            f.writelines(strr)
        # f.close()
        mess = 'success'
    return mess
    
@app.route('/face_login', methods=['POST', 'GET'])
def face_login():
    # 图片上传
    # 人脸检测
    # 人脸特征提取
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp_login." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 只返回检测得分最高的人脸
    l = np.array(output_dict['detection_scores']).argsort()
    i = l[-1]
    if len(output_dict['detection_scores']) > 0 and output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        print(output_dict['detection_scores'][i], x1, y1, x2, y2)
        # 利用obdetection检测出来物体的坐标为相对于图片归一化后的值，0~1
        y1 = int(y1*sp[0])
        x1 = int(x1*sp[1])
        y2 = int(y2*sp[0])
        x2 = int(x2*sp[1])
        im_data = im_data[y1:y2, x1:x2]
        im_data = prewhiten(im_data)
        im_data = cv2.resize(im_data, (160, 160))
        im_data = np.expand_dims(im_data, axis=0)
        emb_1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data, ff_train_placeholder:False})
        emb_1 = emb_1[0]
        # 加载注册人脸 （人脸签到，当人脸数很多时候，加载注册人脸放在face_login外，启动服务时加载/采样搜索引擎-ES）
        with open('face/feature.txt','r') as f:
            fearture_face = f.readlines()
        emb2_str = fearture_face[0].split(',')
        emb_2 = []
        for ss in emb2_str:
            emb_2.append(float(ss))
        emb_2 = np.array(emb_2)
        # 同注册人脸相似性度量：欧氏距离
        dist = np.linalg.norm((emb_1 - emb_2))
        print(dist)
        # 返回度量结果
        if dist < 0.31:
            return 'success'
        else:
            return 'fail'
    return 'fail'

@app.route('/face_distance')
def face_distance():
    im_data1 = read_image('  ')
    im_data1 = np.expand_dims(im_data1, axis=0)
    emb_1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})
    im_data1 = read_image('  ')
    im_data1 = np.expand_dims(im_data1, axis=0)
    emb_2 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})
    distance = np.linalg.norm((emb_1 - emb_2))
    return distance

@app.route('/face_landmark', methods=['POST', 'GET'])
def face_landmark():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    upload_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    
    im_data = cv2.imread(upload_path)
    
    sp = im_data.shape
    
    im_data_re = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data_re, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 只返回检测得分最高的人脸
    l = np.array(output_dict['detection_scores']).argsort()
    i = l[-1]
    if output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        print(output_dict['detection_scores'][i], x1, y1, x2, y2)
        # 利用obdetection检测出来物体的坐标为相对于图片归一化后的值，0~1, 此处还原为原图中的坐标值(x1,y1,x2,y2)
        y1 = int((y1 + (y2 - y1)*0.2)*sp[0])
        x1 = int(x1*sp[1])
        y2 = int(y2*sp[0])
        x2 = int(x2*sp[1])
        
        face_data = im_data[y1:y2, x1:x2]
        cv2.imwrite('face_landmark.jpg', face_data)
        
        face_data = cv2.resize(face_data, (128,128))
    
        pred = face_landmark_sess.run(fl_logits, feed_dict={fl_images_placeholder: np.expand_dims(face_data, axis=0)})
        pred = pred[0]
        
        res = []
        for i in range(0, 136, 2):
            res.append(str((int(pred[i]) * (x2 - x1) + x1) / sp[1]))
            res.append(str((int(pred[i+1]) * (y2 - y1) + y1) / sp[0]))

        # 列表元素转成str
        res = ','.join(res)
        
        return res
    
    return 'error'

# 男变女
@app.route('/face_change0', methods=['POST', 'GET'])
def face_change0():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    upload_path = os.path.join("tmp/tmp_landmark" + time + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    
    im_data = cv2.imread(upload_path)
    
    sp = im_data.shape
    
    im_data_re = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data_re, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 只返回检测得分最高的人脸
    l = np.array(output_dict['detection_scores']).argsort()
    i = l[-1]
    if output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        y1 =   bbox[0]
        x1 =  bbox[1]
        y2 =   (bbox[2])
        x2 = (bbox[3])
        print(output_dict['detection_scores'][i], x1, y1, x2, y2)
        # 利用obdetection检测出来物体的坐标为相对于图片归一化后的值，0~1
        y1 = int((y1 + (y2 - y1)* 0.2)*sp[0])
        x1 = int(x1*sp[1])
        y2 = int(y2*sp[0])
        x2 = int(x2*sp[1])
        
        face_data = im_data[y1:y2, x1:x2]
        face_data = cv2.resize(face_data, (256,256))
        face_men2women = face_change_sess.run(fc_output, {fc_images_placeholder: face_data})
        
        cv2.imwrite('face_change.jpg', face_men2women)
        with  open('face_change.jpg', 'rb') as bites:
            return send_file(
                io.BytesIO(bites.read()),
                attachment_filename='face.jpg',
                mimetype='image/jpg')
    
    return 'error'


# 人脸属性
@app.route('/face_attribute', methods=['POST', 'GET'])
def face_attribute():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    upload_path = os.path.join("tmp/tmp_landmark" + time + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    
    im_data = cv2.imread(upload_path)
    
    sp = im_data.shape
    
    im_data_re = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data_re, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 只返回检测得分最高的人脸
    l = np.array(output_dict['detection_scores']).argsort()
    i = l[-1]
    if output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        y1 =   bbox[0]
        x1 =  bbox[1]
        y2 =   (bbox[2])
        x2 = (bbox[3])
        print(output_dict['detection_scores'][i], x1, y1, x2, y2)
        # 利用obdetection检测出来物体的坐标为相对于图片归一化后的值，0~1
        y1 = int((y1 + (y2 - y1)* 0.2)*sp[0])
        x1 = int(x1*sp[1])
        y2 = int(y2*sp[0])
        x2 = int(x2*sp[1])
        
        face_data = im_data[y1:y2, x1:x2]
        face_data = cv2.resize(face_data, (256,256))
        eyeglasses, young, male, smiling = face_attribute_sess.run([predict_eyeglasses,predict_young,predict_male,predict_smiling], {fa_images_placeholder: np.expand_dims(face_data, axis=0)})
  
        eyeglasses = np.argmax(eyeglasses)
        young = np.argmax(young)
        male = np.argmax(male)
        smiling = np.argmax(smiling)

        print('eyeglasses: {}, young: {}, male: {}, smiling: {}'.format(eyeglasses, young, male, smiling))
        
        return str([eyeglasses, young, male, smiling])
    
    return 'error'


if __name__ == '__main__':
    app.run(host="192.168.1.104", port=90, debug=True)


