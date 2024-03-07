import os
import sys
import time

import cv2
import numpy as np

import utils.utils as utils
from net.inception import InceptionResNetV1
from net.mtcnn import mtcnn


# 重定向输出到空对象
class DevNull:
    def write(self, _):
        pass

    def flush(self):
        pass

class face_rec():
    def __init__(self):
        sys.stdout = DevNull()
        #   创建mtcnn的模型
        #   用于检测人脸
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.6,0.8]
               
        #   载入facenet
        #   将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        face_list = os.listdir("face_dataset")
        self.known_face_encodings=[]
        self.known_face_names=[]
        for face in face_list:
            name = face.split(".")[0]
            img = cv2.imread("./face_dataset/"+face)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #   检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)
            #   转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            #   传入160x160的图片
            #   利用landmark对人脸进行矫正
            rectangle = rectangles[0]
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img, _ = utils.Alignment_1(crop_img,landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
            #   将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            sys.stdout = sys.__stdout__
    def recognize(self,draw):
        # 将标准输出重定向到空对象
        sys.stdout = DevNull()
        start_time = time.time()  # 记录开始时间
        #   人脸识别：先定位，再进行数据库匹配
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        #   检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
        rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

        #   对检测到的人脸进行编码
        face_encodings = []
        for rectangle in rectangles:
            #   截取图像
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #   利用人脸关键点进行人脸对齐
            crop_img,_ = utils.Alignment_1(crop_img,landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding)
            accuracy=1-abs(matches.min())

            accuracy = round(accuracy, 2)
            #matches = [number <= 0.9 for number in matches]


            best_match_index = np.argmin(matches)
            if matches[best_match_index]:
                if accuracy<=0.4:
                    name = "Unknown"
                else:
                    name=self.known_face_names[best_match_index]
            face_names.append(name)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        rectangles = rectangles[:,0:4]
        #   画框
        #   恢复标准输出
        sys.stdout = sys.__stdout__
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 10),  font, 0.9, (0, 255, 0), 2)
            cv2.putText(draw, str(accuracy), (left, top - 10), font, 0.9, (0, 255, 0), 2)
            # 绘制计时信息
            print("Accuracy:{},Elapsed Time:{}".format(accuracy,elapsed_time))

        return draw
        # 在原始帧上绘制人脸框和标签

        #cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)


if __name__ == "__main__":
    dududu = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, draw = video_capture.read()
        dududu.recognize(draw) 
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
