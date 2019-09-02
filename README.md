## 这是什么

一个包含人脸检测、特征提取、人脸关键点定位、人脸属性、人脸性别变换的微信小程序后端脚本。

## 需要什么

tensorflow == 1.12

numpy ==  1.14.5

opencv == 3.4.1

flask == 0.12.2

## 如何食用

下载训练好的 pb 文件，放入model文件夹下

[人脸特征提取](https://pan.baidu.com/s/1_-e8SdHDqeXZ_rXHUKBotg )      提取码：0iga

[人脸框检测](https://pan.baidu.com/s/1OrV1a5UyVapLCg-MNI2xcw)          提取码：zt5y

[人脸属性](https://pan.baidu.com/s/1RxD9H1-Vkf7D6fXP69AEAw )              提取码：rd86

[人脸关键点](https://pan.baidu.com/s/10sXmMFK19I8xPrE7Q5d1tg )          提取码：umoa

[性别变换-men2women](https://pan.baidu.com/s/1BfSvNAuxNKUPJJ3m1YhRNA)      提取码：lxbn

[性别变换-women2men](https://pan.baidu.com/s/1j9DgskeJowgC5omJRnmlbw)       提取码：iq8b

修改 Deep_face_backend.py 中地址和端口

运行 Deep_face_backend.py 文件

## 都返回了啥

face_detect：输入含有人脸的图片，返回检测到人脸框的坐标值 

face_feature:  输入人脸框图，返回人脸特征，主要做为后续任务的特征提取

face_register：基于face_feature提取的人脸特征对用户人脸进行注册

face_login：基于face_feature提取的人脸特征与库中人脸特征进行比对登陆 

face_landmark：输入人脸框，定位人脸关键点，返回68个关键点的坐标，136维度

face_change0：输入含有人脸的图片，返回性别变换后人脸图片

face_attribute：输入含有人脸的图片，返回4个属性（是否戴眼镜、年轻否、性别、表情）

## 有啥用

老实说，没啥用。

老实说，微信小程序前端写的太差，就不放上来。

老实说，只是交流一个思路。

## 感谢

会写代码的厨师