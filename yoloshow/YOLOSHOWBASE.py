import random

from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 rtdetr "
                          "yolov5-seg yolov8-seg yolov11-seg yolov8-pose yolov8-obb yolov11-obb yolov11-pose "
                          "fastsam sam samv2")
from utils.logger import LoggerUtils
import re
import socket
from urllib.parse import urlparse, urlunparse
import torch
import json
import os
import shutil
import cv2
import numpy as np
from ui.utils.AcrylicFlyout import AcrylicFlyoutView, AcrylicFlyout
from ui.utils.TableView import TableViewQWidget
from ui.utils.drawFigure import PlottingThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFileDialog, QGraphicsDropShadowEffect, QFrame, QPushButton
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QPoint, QTimer
from qfluentwidgets import RoundMenu, MenuAnimationType, Action
import importlib
from ui.utils.rtspDialog import CustomMessageBox
from models import common, yolo, experimental
from ui.utils.webCamera import Camera, WebcamThread
from yolocode.yolov5.YOLOv5Thread import YOLOv5Thread
from yolocode.yolov7.YOLOv7Thread import YOLOv7Thread
from yolocode.yolov8.YOLOv8Thread import YOLOv8Thread
from yolocode.yolov9.YOLOv9Thread import YOLOv9Thread
from yolocode.yolov5.YOLOv5SegThread import YOLOv5SegThread
from yolocode.yolov8.YOLOv8SegThread import YOLOv8SegThread
from yolocode.rtdetr.RTDETRThread import RTDETRThread
from yolocode.yolov8.YOLOv8PoseThread import YOLOv8PoseThread
from yolocode.yolov8.YOLOv8ObbThread import YOLOv8ObbThread
from yolocode.yolov10.YOLOv10Thread import YOLOv10Thread
from yolocode.yolov11.YOLOv11Thread import YOLOv11Thread
from yolocode.yolov11.YOLOv11SegThread import YOLOv11SegThread
from yolocode.yolov11.YOLOv11ObbThread import YOLOv11ObbThread
from yolocode.yolov11.YOLOv11PoseThread import YOLOv11PoseThread
from yolocode.fastsam.FastSAMThread import FastSAMThread
from yolocode.sam.SAMThread import SAMThread
from yolocode.sam.SAMv2Thread import SAMv2Thread

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_SETTING_BAR = 300
WIDTH_LOGO = 60
WINDOW_SPLIT_BODY = 20
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']
# 模型名称和线程类映射
MODEL_THREAD_CLASSES = {
    "yolov5": YOLOv5Thread,
    "yolov7": YOLOv7Thread,
    "yolov8": YOLOv8Thread,
    "yolov9": YOLOv9Thread,
    "yolov10": YOLOv10Thread,
    "yolov11": YOLOv11Thread,
    "rtdetr": RTDETRThread,
    "yolov5-seg": YOLOv5SegThread,
    "yolov8-seg": YOLOv8SegThread,
    "yolov11-seg": YOLOv11SegThread,
    "yolov8-pose": YOLOv8PoseThread,
    "yolov11-pose": YOLOv11PoseThread,
    "yolov8-obb": YOLOv8ObbThread,
    "yolov11-obb": YOLOv11ObbThread,
    "fastsam": FastSAMThread,
    "sam": SAMThread,
    "samv2": SAMv2Thread,
    "mobilenet": YOLOv8Thread,  # MobileNet uses YOLOv8 detection thread since it's compatible
    "best": YOLOv10Thread,  # best.pt is a YOLOv10 trash detection model 
    "yolov10n": YOLOv10Thread   # yolov10n.pt is a YOLOv10 trash detection model
}
# 扩展MODEL_THREAD_CLASSES字典
MODEL_NAME_DICT = list(MODEL_THREAD_CLASSES.items())
for key, value in MODEL_NAME_DICT:
    MODEL_THREAD_CLASSES[f"{key}_left"] = value
    MODEL_THREAD_CLASSES[f"{key}_right"] = value

ALL_MODEL_NAMES = ["yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov5-seg", "yolov8-seg", "rtdetr",
                   "yolov8-pose", "yolov8-obb", "fastsam", "sam", "samv2", "mobilenet"]
loggertool = LoggerUtils()


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOWBASE:
    def __init__(self):
        super().__init__()
        self.inputPath = None
        self.yolo_threads = None
        self.result_statistic = None
        self.detect_result = None
        self.allModelNames = ALL_MODEL_NAMES

    # 初始化左侧菜单栏
    def initSiderWidget(self):
        # --- 侧边栏 --- #
        self.ui.leftBox.setFixedWidth(WIDTH_LEFT_BOX_STANDARD)
        # logo
        self.ui.logo.setFixedSize(WIDTH_LOGO, WIDTH_LOGO)

        # 将左侧菜单栏的按钮固定宽度
        for child_left_box_widget in self.ui.leftbox_bottom.children():

            if isinstance(child_left_box_widget, QFrame):
                child_left_box_widget.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

                for child_left_box_widget_btn in child_left_box_widget.children():
                    if isinstance(child_left_box_widget_btn, QPushButton):
                        child_left_box_widget_btn.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

    # 加载模型
    def initModel(self, yoloname=None):
        thread = self.yolo_threads.get(yoloname)
        if not thread:
            raise ValueError(f"No thread found for '{yoloname}'")
        thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        thread.progress_value = self.ui.progress_bar.maximum()

        # 信号槽连接使用单独定义的函数，减少闭包的创建
        thread.send_input.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
        thread.send_msg.connect(lambda x: self.showStatus(x))
        thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
        thread.send_fps.connect(lambda x: self.ui.fps_label.setText(str(x)))
        thread.send_class_num.connect(lambda x: self.ui.Class_num.setText(str(x)))
        thread.send_target_num.connect(lambda x: self.ui.Target_num.setText(str(x)))
        thread.send_result_picture.connect(lambda x: self.setResultStatistic(x))
        thread.send_result_table.connect(lambda x: self.setTableResult(x))

    # 阴影效果
    def shadowStyle(self, widget, Color, top_bottom=None):
        shadow = QGraphicsDropShadowEffect(self)
        if 'top' in top_bottom and 'bottom' not in top_bottom:
            shadow.setOffset(0, -5)
        elif 'bottom' in top_bottom and 'top' not in top_bottom:
            shadow.setOffset(0, 5)
        else:
            shadow.setOffset(5, 5)
        shadow.setBlurRadius(10)  # 阴影半径
        shadow.setColor(Color)  # 阴影颜色
        widget.setGraphicsEffect(shadow)

    # 侧边栏缩放
    def scaleMenu(self):
        # standard = 80
        # maxExtend = 180

        leftBoxStart = self.ui.leftBox.width()
        _IS_EXTENDED = leftBoxStart == WIDTH_LEFT_BOX_EXTENDED

        if _IS_EXTENDED:
            leftBoxEnd = WIDTH_LEFT_BOX_STANDARD
        else:
            leftBoxEnd = WIDTH_LEFT_BOX_EXTENDED

        # animation
        self.animation = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.animation.setDuration(500)  # ms
        self.animation.setStartValue(leftBoxStart)
        self.animation.setEndValue(leftBoxEnd)
        self.animation.setEasingCurve(QEasingCurve.InOutQuint)
        self.animation.start()

    # 设置栏缩放
    def scalSetting(self):
        # GET WIDTH
        widthSettingBox = self.ui.settingBox.width()  # right set column width
        widthLeftBox = self.ui.leftBox.width()  # left column length
        maxExtend = WIDTH_SETTING_BAR
        standard = 0

        # SET MAX WIDTH
        if widthSettingBox == 0:
            widthExtended = maxExtend
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                    background-color: qlineargradient(x1:0, y1:0, x2:1 , y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
                                }
                              """)
        else:
            widthExtended = standard
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                }
                              """)

        # ANIMATION LEFT BOX
        self.left_box = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.left_box.setDuration(500)
        self.left_box.setStartValue(widthLeftBox)
        self.left_box.setEndValue(68)
        self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

        # ANIMATION SETTING BOX
        self.setting_box = QPropertyAnimation(self.ui.settingBox, b"minimumWidth")
        self.setting_box.setDuration(500)
        self.setting_box.setStartValue(widthSettingBox)
        self.setting_box.setEndValue(widthExtended)
        self.setting_box.setEasingCurve(QEasingCurve.InOutQuart)

        # SET QSS Change
        self.qss_animation = QPropertyAnimation(self.ui.mainbox, b"styleSheet")
        self.qss_animation.setDuration(300)
        self.qss_animation.setStartValue("""
            QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
            }
        """)
        self.qss_animation.setEndValue("""
             QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
            }
        """)
        self.qss_animation.setEasingCurve(QEasingCurve.InOutQuart)

        # GROUP ANIMATION
        self.group = QParallelAnimationGroup()
        self.group.addAnimation(self.left_box)
        self.group.addAnimation(self.setting_box)
        self.group.start()

    # 最大化最小化窗口
    def maxorRestore(self):
        global GLOBAL_WINDOW_STATE
        status = GLOBAL_WINDOW_STATE
        if status:
            # 获取当前屏幕的宽度和高度
            self.showMaximized()
            self.ui.maximizeButton.setStyleSheet("""
                          QPushButton:hover{
                               background-color:rgb(139, 29, 31);
                               border-image: url(:/leftbox/images/newsize/scalling.png);
                           }
                      """)
            GLOBAL_WINDOW_STATE = False
        else:
            self.showNormal()
            self.ui.maximizeButton.setStyleSheet("""
                                      QPushButton:hover{
                                           background-color:rgb(139, 29, 31);
                                           border-image: url(:/leftbox/images/newsize/max.png);
                                       }
                                  """)
            GLOBAL_WINDOW_STATE = True

    # 选择照片/视频 并展示
    def selectFile(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/file.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_path = config['file_path']
        if not os.path.exists(file_path):
            file_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your Image / Video",  # 标题
            file_path,  # 默认打开路径为当前路径
            "Image / Video type (*.jpg *.jpeg *.png *.bmp *.dib *.jpe *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            self.inputPath = file
            glo.set_value('inputPath', self.inputPath)
            # 如果是视频， 显示第一帧
            if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                # 显示第一帧
                self.cap = cv2.VideoCapture(self.inputPath)
                ret, frame = self.cap.read()
                if ret:
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.showImg(frame, self.ui.main_leftbox, 'img')
            # 如果是图片 正常显示
            else:
                self.showImg(self.inputPath, self.ui.main_leftbox, 'path')
            self.showStatus('Loaded File：{}'.format(os.path.basename(self.inputPath)))
            config['file_path'] = os.path.dirname(self.inputPath)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 选择摄像头
    def selectWebcam(self):
        try:
            # get the number of local cameras
            cam_num, cams = Camera().get_cam_num()
            if cam_num > 0:
                popMenu = RoundMenu(parent=self)
                popMenu.setFixedWidth(self.ui.leftbox_bottom.width())
                actions = []

                for cam in cams:
                    cam_name = f'Camera_{cam}'
                    actions.append(Action(cam_name))
                    popMenu.addAction(actions[-1])
                    actions[-1].triggered.connect(lambda: self.actionWebcam(cam))

                x = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).x()
                y = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).y()
                y = y - self.ui.webcamBox.frameGeometry().height() * 2
                pos = QPoint(x, y)
                popMenu.exec(pos, aniType=MenuAnimationType.DROP_DOWN)
            else:
                self.showStatus('No camera found !!!')
        except Exception as e:
            self.showStatus('%s' % e)

    # 调用网络摄像头
    def actionWebcam(self, cam):
        self.showStatus(f'Loading camera：Camera_{cam}')
        self.thread = WebcamThread(cam)
        self.thread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        self.thread.start()
        self.inputPath = int(cam)
        
        # Send a Twilio message indicating trash detection
        try:
            from utils.twilio_utils import send_direct_sms
            send_direct_sms("Trash has been detected")
            self.showStatus("Twilio message sent: Trash detected")
        except Exception as e:
            self.showStatus(f"Failed to send Twilio message: {str(e)}")

    # 选择文件夹
    def selectFolder(self):
        config_file = f'{self.current_workpath}/config/folder.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        folder_path = config['folder_path']
        if not os.path.exists(folder_path):
            folder_path = os.getcwd()
        FolderPath = QFileDialog.getExistingDirectory(
            self,
            "Select your Folder",
            folder_path  # 起始目录
        )
        if FolderPath:
            FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            # self.yolov5_thread.source = Foldername
            self.inputPath = Foldername
            self.showStatus('Loaded Folder：{}'.format(os.path.basename(FolderPath)))
            config['folder_path'] = FolderPath
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 选择网络摄像头 Rtsp
    def selectRtsp(self):
        # rtsp://rtsp-test-server.viomic.com:554/stream
        rtspDialog = CustomMessageBox(self, mode="single")
        self.rtspUrl = None
        if rtspDialog.exec():
            self.rtspUrl = rtspDialog.urlLineEdit.text()
            if rtspDialog.authCheckBox.isChecked() and self.rtspUrl:
                parsed_url = urlparse(self.rtspUrl)
                # 构造新 netloc，包含用户名和密码
                username = rtspDialog.usernameLineEdit.text()
                password = rtspDialog.passwordLineEdit.text()
                netloc = f"{username}:{password}@{parsed_url.hostname}:{parsed_url.port}"
                # 重组 URL
                self.rtspUrl = urlunparse(
                    (parsed_url.scheme, netloc, parsed_url.path, parsed_url.params, parsed_url.query,
                     parsed_url.fragment))

        if self.rtspUrl:
            parsed_url = urlparse(self.rtspUrl)
            if parsed_url.scheme == 'rtsp':
                if not self.checkRtspUrl(self.rtspUrl):
                    self.showStatus('Rtsp stream is not available')
                    return False
                self.showStatus(f'Loading Rtsp：{self.rtspUrl}')
                self.rtspThread = WebcamThread(self.rtspUrl)
                self.rtspThread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
                self.rtspThread.start()
                self.inputPath = self.rtspUrl
            elif parsed_url.scheme in ['http', 'https']:
                if not self.checkHttpUrl(self.rtspUrl):
                    self.showStatus('Http stream is not available')
                    return False
                self.showStatus(f'Loading Http：{self.rtspUrl}')
                self.rtspThread = WebcamThread(self.rtspUrl)
                self.rtspThread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
                self.rtspThread.start()
                self.inputPath = self.rtspUrl
            else:
                self.showStatus('URL is not an rtsp stream')
                return False

    # 检测网络摄像头 Rtsp 是否连通
    def checkRtspUrl(self, url, timeout=5):
        try:
            # 解析URL获取主机名和端口
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 554  # RTSP默认端口是554

            # 创建socket对象
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # 尝试连接
            sock.connect((hostname, port))
            # 关闭socket
            sock.close()
            return True
        except Exception:
            return False

    # 检测Http网络摄像头 是否连通
    def checkHttpUrl(self, url, timeout=5):
        try:
            # 解析URL获取主机名和端口
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 80  # HTTP默认端口是80

            # 创建socket对象
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # 尝试连接
            sock.connect((hostname, port))
            # 关闭socket
            sock.close()
            return True
        except Exception as e:
            return False

    # 显示Label图片
    def showImg(self, img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    # resize 窗口大小
    def resizeGrip(self):
        self.left_grip.setGeometry(0, 10, 10, self.height())
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 导入模块
    def importModel(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/model.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.model_path = config['model_path']
        if not os.path.exists(self.model_path):
            self.model_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your YOLO Model",  # 标题
            self.model_path,  # 默认打开路径为当前路径
            "Model File (*.pt)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            fileptPath = os.path.join(self.pt_Path, os.path.basename(file))
            if not os.path.exists(fileptPath):
                shutil.copy(file, self.pt_Path)
                self.showStatus('Loaded Model：{}'.format(os.path.basename(file)))
                config['model_path'] = os.path.dirname(file)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            else:
                self.showStatus('Model already exists')

    # 查看当前模型
    def checkCurrentModel(self, mode=None):
        # Define model and condition mapping
        model_conditions = {
            "yolov5": lambda name: "yolov5" in name and not self.checkSegName(name) and not "trash" in name.lower(),
            "yolov7": lambda name: "yolov7" in name and not "trash" in name.lower(),
            "yolov8": lambda name: "yolov8" in name and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]) and not "trash" in name.lower(),
            "yolov9": lambda name: "yolov9" in name and not "trash" in name.lower(),
            "yolov10": lambda name: ("yolov10" in name or name == "yolov10.pt") and not "trash" in name.lower(),
            "yolov11": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]) and not "trash" in name.lower(),
            "rtdetr": lambda name: "rtdetr" in name and not "trash" in name.lower(),
            "yolov5-seg": lambda name: "yolov5" in name and self.checkSegName(name) and not "trash" in name.lower(),
            "yolov8-seg": lambda name: "yolov8" in name and self.checkSegName(name) and not "trash" in name.lower(),
            "yolov11-seg": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkSegName(name) and not "trash" in name.lower(),
            "yolov8-pose": lambda name: "yolov8" in name and self.checkPoseName(name) and not "trash" in name.lower(),
            "yolov11-pose": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkPoseName(name) and not "trash" in name.lower(),
            "yolov8-obb": lambda name: "yolov8" in name and self.checkObbName(name) and not "trash" in name.lower(),
            "yolov11-obb": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkObbName(name) and not "trash" in name.lower(),
            "fastsam": lambda name: "fastsam" in name and not "trash" in name.lower(),
            "samv2": lambda name: any(sub in name for sub in ["sam2", "samv2"]) and not "trash" in name.lower(),
            "sam": lambda name: "sam" in name and not "trash" in name.lower(),
            "mobilenet": lambda name: "mobilenet" in name or name == "mobilenet_trash_detector.pt" or "trash" in name.lower(),
            "yolov10n": lambda name: name == "yolov10.pt" # Use yolov10.pt instead of yolov10n.pt or best.pt
        }

        if mode:
            # VS mode
            model_name = self.model_name if hasattr(self, 'model_name') else ""
            model_name = model_name.lower()
            
            # Special handling for yolov10.pt
            if model_name == "yolov10.pt":
                return f"yolov10n_{mode}"
                
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return f"{yoloname}_{mode}"
        else:
            # Single mode
            model_name = self.model_name.lower() if hasattr(self, 'model_name') else ""
            
            # Special handling for yolov10.pt
            if model_name == "yolov10.pt":
                return "yolov10n"
                
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return yoloname
                    
        # Fallback special case handling if conditions above fail
        if mode:
            model_name = self.model_name if hasattr(self, 'model_name') else ""
            model_name = model_name.lower()
            if "mobilenet_trash_detector.pt" in model_name:
                return f"mobilenet_{mode}"
        else:
            model_name = self.model_name.lower() if hasattr(self, 'model_name') else ""
            if "mobilenet_trash_detector.pt" in model_name:
                return "mobilenet"
                
        return None

    # 检查模型是否符合命名要求
    def checkModelName(self, modelname):
        for name in self.allModelNames:
            if modelname in name:
                return True
        return False

    def checkTaskName(self, modelname, taskname):
        if "yolov5" in modelname:
            return bool(re.match(f'yolo.?5.?-{taskname}.*\.pt$', modelname))
        elif "yolov7" in modelname:
            return bool(re.match(f'yolo.?7.?-{taskname}.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(f'yolo.?8.?-{taskname}.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(f'yolo.?9.?-{taskname}.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(f'yolo.?10.?-{taskname}.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(f'yolo.?11.?-{taskname}.*\.pt$', modelname))

    # 解决 Modelname 当中的 seg命名问题
    def checkSegName(self, modelname):
        return self.checkTaskName(modelname, "seg")

    # 解决  Modelname 当中的 pose命名问题
    def checkPoseName(self, modelname):
        return self.checkTaskName(modelname, "pose")

    # 解决  Modelname 当中的 pose命名问题
    def checkObbName(self, modelname):
        return self.checkTaskName(modelname, "obb")

    # 停止运行中的模型
    def quitRunningModel(self, stop_status=False):
        for yolo_name in self.yolo_threads.threads_pool.keys():
            try:
                if stop_status:
                    self.yolo_threads.get(yolo_name).stop_dtc = True
                self.yolo_threads.stop_thread(yolo_name)
            except Exception as err:
                loggertool.info(f"Error: {err}")

    # 在MessageBar显示消息
    def showStatus(self, msg):
        self.ui.message_bar.setText(msg)
        if msg == 'Finish Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.progress_bar.setValue(0)
            self.ui.save_status_button.setEnabled(True)
        elif msg == 'Stop Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.save_status_button.setEnabled(True)
            self.ui.progress_bar.setValue(0)
            self.ui.main_leftbox.clear()  # clear image display
            self.ui.main_rightbox.clear()
            self.ui.Class_num.setText('--')
            self.ui.Target_num.setText('--')
            self.ui.fps_label.setText('--')

    # 导出结果状态判断
    def saveStatus(self):
        if self.ui.save_status_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run image results are not saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_res = False
            self.ui.save_button.setEnabled(False)
        elif self.ui.save_status_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run image results will be saved.')
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.save_res = True
            self.ui.save_button.setEnabled(True)

    # 导出检测结果 --- 过程代码
    def saveResultProcess(self, outdir, current_model_name, folder):
        yolo_thread = self.yolo_threads.get(current_model_name)
        if folder:
            try:
                output_dir = os.path.dirname(yolo_thread.res_path)
                if not os.path.exists(output_dir):
                    self.showStatus('Please wait for the result to be generated')
                    return
                for filename in os.listdir(output_dir):
                    source_path = os.path.join(output_dir, filename)
                    destination_path = os.path.join(outdir, filename)
                    if os.path.isfile(source_path):
                        shutil.copy(source_path, destination_path)
                self.showStatus('Saved Successfully in {}'.format(outdir))
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")
        else:
            try:
                if not os.path.exists(yolo_thread.res_path):
                    self.showStatus('Please wait for the result to be generated')
                    return
                shutil.copy(yolo_thread.res_path, outdir)
                self.showStatus('Saved Successfully in {}'.format(outdir))
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")

    def loadAndSetParams(self, config_file, params):
        if not os.path.exists(config_file):
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                params.update(json.load(f))
        return params

    # 加载 Setting 栏
    def loadConfig(self):
        # 1、随机初始化超参数，防止切换模型时，超参数不变
        params = {"iou": round(random.uniform(0, 1), 2),
                  "conf": round(random.uniform(0, 1), 2),
                  "delay": random.randint(10, 50),
                  "line_thickness": random.randint(1, 5)}
        self.updateParams(params)
        
        # 2、绑定配置项超参数
        params = {"iou": 0.45, "conf": 0.25, "delay": 10, "line_thickness": 3}
        
        # Check if we're using a trash detection model and adjust confidence threshold
        model_name = self.model_name.lower() if hasattr(self, 'model_name') else ""
        if model_name and ("trash" in model_name or model_name == "yolov10.pt" or model_name == "mobilenet_trash_detector.pt"):
            params["conf"] = 0.15  # Lower confidence threshold for trash detection models
            self.showStatus(f"Using lower confidence threshold ({params['conf']}) for trash detection model")
            
        params = self.loadAndSetParams('config/setting.json', params)
        self.updateParams(params)
        
        # Check Twilio configuration status
        try:
            from utils.twilio_utils import load_twilio_config
            config = load_twilio_config()
            enabled = config.get("enabled", False)
            has_auth = bool(config.get("auth_token", "").strip())
            
            status = "enabled" if enabled else "disabled"
            auth_status = "configured" if has_auth else "missing auth token"
            self.showStatus(f"Twilio SMS notifications are {status} ({auth_status})")
            
            # If auth token is missing but feature is enabled, show a clear warning
            if enabled and not has_auth:
                self.showStatus("WARNING: Twilio SMS is enabled but auth token is missing. SMS notifications will not work.")
                self.showStatus("Update config/twilio_config.json with your Twilio auth token.")
        except Exception as e:
            self.showStatus(f"Error checking Twilio configuration: {str(e)}")
            
        # Apply settings to active thread
        current_model_name = self.checkCurrentModel()
        if current_model_name is not None:
            yolo_thread = self.yolo_threads.get(current_model_name)
            if yolo_thread is not None:
                yolo_thread.iou_thres = params["iou"]
                yolo_thread.conf_thres = params["conf"]
                yolo_thread.speed_thres = params["delay"]
                yolo_thread.line_thickness = params["line_thickness"]

    # 更新Config超参数
    def updateParams(self, params):
        self.ui.iou_spinbox.setValue(params['iou'])
        self.ui.iou_slider.setValue(int(params['iou'] * 100))
        self.ui.conf_spinbox.setValue(params['conf'])
        self.ui.conf_slider.setValue(int(params['conf'] * 100))
        self.ui.speed_spinbox.setValue(params['delay'])
        self.ui.speed_slider.setValue(params['delay'])
        self.ui.line_spinbox.setValue(params['line_thickness'])
        self.ui.line_slider.setValue(params['line_thickness'])

    # 加载 pt 模型到 model_box
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box.clear()
            self.ui.model_box.addItems(self.pt_list)

    # 重载模型
    def reloadModel(self):
        importlib.reload(common)
        importlib.reload(yolo)
        importlib.reload(experimental)

    # 调整超参数
    def changeValue(self, x, flag):
        if flag == 'iou_spinbox':
            self.ui.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.ui.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.showStatus('IOU Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.ui.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.ui.conf_spinbox.setValue(x / 100)
            self.showStatus('Conf Threshold: %s' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.ui.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.ui.speed_spinbox.setValue(x)
            self.showStatus('Delay: %s ms' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.speed_thres = x  # ms
        elif flag == 'line_spinbox':
            self.ui.line_slider.setValue(x)
        elif flag == 'line_slider':
            self.ui.line_spinbox.setValue(x)
            self.showStatus('Line Width: %s' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.line_thickness = x

    # 修改YOLOv5、YOLOv7、YOLOv9 解决 yolo.py冲突
    def solveYoloConflict(self, ptnamelst):
        for ptname in ptnamelst:
            ptbaseName = os.path.basename(ptname)
            if "yolov5" in ptbaseName and not self.checkSegName(ptbaseName):
                glo.set_value('yoloname', "yolov5")
                self.reloadModel()
                from models.yolo import Detect_YOLOV5
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov5_detect = Detect_YOLOV5(anchors=_anchors, nc=_nc, ch=_ch)
                        yolov5_detect.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov5_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov5_detect
                torch.save(net, ptname)
            elif "yolov5" in ptbaseName and self.checkSegName(ptbaseName):
                glo.set_value('yoloname', "yolov5-seg")
                self.reloadModel()
                from models.yolo import Segment_YOLOV5
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Segment':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_seg_layers = _yaml_lst[-1][0]
                        for layer in _yaml_seg_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov5_seg = Segment_YOLOV5(anchors=_anchors, nc=_nc, ch=_ch)
                        _module.detect = yolov5_seg.detect
                        yolov5_seg.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov5_seg.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov5_seg
                torch.save(net, ptname)
            elif "yolov7" in ptbaseName:
                glo.set_value('yoloname', "yolov7")
                self.reloadModel()
                from models.yolo import Detect_YOLOV7
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov7_detect = Detect_YOLOV7(anchors=_anchors, nc=_nc, ch=_ch)
                        yolov7_detect.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov7_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov7_detect
                torch.save(net, ptname)
            elif "yolov9" in ptbaseName:
                glo.set_value('yoloname', "yolov9")
                self.reloadModel()
                from models.yolo import Detect_YOLOV9
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _nc = _module.nc
                        yolov9_detect = Detect_YOLOV9(nc=_nc, ch=_ch)
                        for _new_param, _old_param in zip(yolov9_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov9_detect
                torch.save(net, ptname)
        glo.set_value("yoloname", "yolov5 yolov7 yolov8 yolov9 yolov5-seg yolov8-seg rtdetr yolov8-pose")
        self.reloadModel()

    # 接受统计结果，然后写入json中
    def setResultStatistic(self, value):
        # 写入 JSON 文件
        with open('config/result.json', 'w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=4)
        # --- 获取统计结果 + 绘制柱状图 --- #
        self.result_statistic = value
        self.plot_thread = PlottingThread(self.result_statistic, self.current_workpath)
        self.plot_thread.start()
        # --- 获取统计结果 + 绘制柱状图 --- #

    # 展示柱状图结果
    def showResultStatics(self):
        self.resutl_statistic = dict()
        # 读取 JSON 文件
        with open(self.current_workpath + r'\config\result.json', 'r', encoding='utf-8') as file:
            self.result_statistic = json.load(file)
        if self.result_statistic:
            # 创建新字典，使用中文键
            result_str = ""
            for index, (key, value) in enumerate(self.result_statistic.items()):
                result_str += f"{key}:{value}x \t"
                if (index + 1) % 4 == 0:
                    result_str += "\n"

            view = AcrylicFlyoutView(
                title='Detected Target Category Distribution (Percentage)',
                content=result_str,
                image=self.current_workpath + r'\config\result.png',
                isClosable=True
            )

        else:
            view = AcrylicFlyoutView(
                title='Result Statistics',
                content="No completed target detection results detected, please execute the detection task first!",
                isClosable=True
            )

        # 修改字体大小
        view.titleLabel.setStyleSheet("""font-size: 30px; 
                                            color: black; 
                                            font-weight: bold; 
                                            font-family: 'KaiTi';
                                        """)
        view.contentLabel.setStyleSheet("""font-size: 25px; 
                                            color: black; 
                                            font-family: 'KaiTi';""")
        # 修改image的大小
        width = self.ui.rightbox_main.width() // 2.5
        height = self.ui.rightbox_main.height() // 2.5
        view.imageLabel.setFixedSize(width, height)
        # adjust layout (optional)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)

        # show view
        w = AcrylicFlyout.make(view, self.ui.rightbox_play, self)
        view.closed.connect(w.close)

    # 获取表格结果的列表
    def setTableResult(self, value):
        self.detect_result = value

    # 展示表格结果
    def showTableResult(self):
        self.table_result = TableViewQWidget(infoList=self.detect_result)
        self.table_result.show()

    # 切换Twilio SMS功能
    def toggleTwilioSMS(self, enabled=None):
        try:
            from utils.twilio_utils import load_twilio_config
            import json
            
            config_path = "config/twilio_config.json"
            
            # Load current config
            config = load_twilio_config(config_path)
            
            # If enabled parameter is provided, use it; otherwise toggle current state
            if enabled is not None:
                config["enabled"] = enabled
            else:
                config["enabled"] = not config.get("enabled", False)
                
            # Save updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
                
            # Update UI and show status
            status = "enabled" if config["enabled"] else "disabled"
            self.showStatus(f"Twilio SMS notifications are now {status}")
            
            # Update global setting
            from utils import glo
            glo.set_value("twilio_enabled", config["enabled"])
            
            # Update thread settings
            if hasattr(self, "yolo_threads"):
                for name, thread in self.yolo_threads.threads_pool.items():
                    if hasattr(thread, "twilio_enabled"):
                        thread.twilio_enabled = config["enabled"]
                        
            return config["enabled"]
        except Exception as e:
            self.showStatus(f"Error toggling Twilio SMS: {str(e)}")
            return False

    # Function to send a Twilio message
    # Ensure this function is defined in the class or imported from utils.twilio_utils
    # def sendTwilioMessage(self, message):
    #     from utils.twilio_utils import send_message
    #     try:
    #         send_message(message)
    #         self.showStatus("Twilio message sent successfully.")
    #     except Exception as e:
    #         self.showStatus(f"Failed to send Twilio message: {str(e)}")
