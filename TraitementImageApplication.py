from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore
import PyQt5
import cv2
import math
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter,median_filter
import pywt
import heapq
from collections import defaultdict

from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from skimage import io

# import imutils
import sys
from sklearn.cluster import KMeans

class User_Interface(object):
        def setupUi(self, MainWindow):
                if not MainWindow.objectName():                       
                        MainWindow.setObjectName(u"MainWindow")
                        # MainWindow.setMinimumSize(1600, 827)
                        MainWindow.showMaximized()
                        font = QFont()
                        font.setPointSize(12)
                        MainWindow.setFont(font)
                        MainWindow.setAutoFillBackground(False)
                        MainWindow.setStyleSheet(u"background-color:#b5b8b6;")
                        MainWindow.setTabShape(QTabWidget.Triangular)
                        self.centralwidget = QWidget(MainWindow)
                        self.centralwidget.setObjectName(u"centralwidget")
                        self.widgetNavBar = QWidget(self.centralwidget)
                        self.widgetNavBar.setObjectName(u"NavBar")
                        self.widgetNavBar.setGeometry(QRect(0, 0, 1600, 60))
                        self.widgetNavBar.setStyleSheet(u"background-color:#282d36;")
                        self.newImageBtn = QToolButton(self.widgetNavBar)
                        self.newImageBtn.setObjectName(u"newImageBtn")
                        self.newImageBtn.setGeometry(QRect(20, 10, 70, 50))
                        font1 = QFont()
                        font1.setFamilies([u"Arial"])
                        font1.setPointSize(10)
                        font1.setBold(False)
                        self.newImageBtn.setFont(font1)
                        self.newImageBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.newImageBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon = QIcon()
                        icon.addFile(u"imagesForUI\\add-photo.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.newImageBtn.setIcon(icon)
                        self.newImageBtn.setIconSize(QSize(30, 30))
                        self.newImageBtn.setCheckable(False)
                        self.newImageBtn.setPopupMode(QToolButton.InstantPopup)
                        self.newImageBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.newImageBtn.setAutoRaise(False)
                        self.saveImageBtn = QToolButton(self.widgetNavBar)
                        self.saveImageBtn.setObjectName(u"saveImage")
                        self.saveImageBtn.setGeometry(QRect(90, 10, 70, 50))
                        font2 = QFont()
                        font2.setPointSize(10)
                        font2.setBold(False)
                        self.saveImageBtn.setFont(font2)
                        self.saveImageBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.saveImageBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon1 = QIcon()
                        icon1.addFile(u"imagesForUI\\download-file.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.saveImageBtn.setIcon(icon1)
                        self.saveImageBtn.setIconSize(QSize(30, 30))
                        self.saveImageBtn.setCheckable(False)
                        self.saveImageBtn.setPopupMode(QToolButton.InstantPopup)
                        self.saveImageBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.saveImageBtn.setAutoRaise(False)
                        self.closeBtn = QToolButton(self.widgetNavBar)
                        self.closeBtn.setObjectName(u"close")
                        self.closeBtn.setGeometry(QRect(1510, 10, 70, 50))
                        self.closeBtn.setFont(font2)
                        self.closeBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.closeBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon3 = QIcon()
                        icon3.addFile(u"imagesForUI\\close.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.closeBtn.setIcon(icon3)
                        self.closeBtn.setIconSize(QSize(30, 30))
                        self.closeBtn.setCheckable(False)
                        self.closeBtn.setPopupMode(QToolButton.InstantPopup)
                        self.closeBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        # self.closeBtn.setAutoRaise(False)
                        self.OrigineBtn = QToolButton(self.widgetNavBar)
                        self.OrigineBtn.setObjectName(u"Origine")
                        self.OrigineBtn.setGeometry(QRect(1440, 10, 70, 50))
                        self.OrigineBtn.setFont(font2)
                        self.OrigineBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.OrigineBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon14 = QIcon()
                        icon14.addFile(u"imagesForUI\\reply-all.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.OrigineBtn.setIcon(icon14)
                        self.OrigineBtn.setIconSize(QSize(30, 30))
                        self.OrigineBtn.setCheckable(False)
                        self.OrigineBtn.setPopupMode(QToolButton.InstantPopup)
                        self.OrigineBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.OrigineBtn.setAutoRaise(False)
                        self.rotateBtn = QToolButton(self.widgetNavBar)
                        self.rotateBtn.setObjectName(u"rotate")
                        self.rotateBtn.setGeometry(QRect(660, 10, 70, 50))
                        self.rotateBtn.setFont(font2)
                        self.rotateBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.rotateBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon5 = QIcon()
                        icon5.addFile(u"imagesForUI\\refresh.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.rotateBtn.setIcon(icon5)
                        self.rotateBtn.setIconSize(QSize(30, 30))
                        self.rotateBtn.setCheckable(False)
                        self.rotateBtn.setPopupMode(QToolButton.InstantPopup)
                        self.rotateBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.rotateBtn.setAutoRaise(False)
                        self.resizeBtn = QToolButton(self.widgetNavBar)
                        self.resizeBtn.setObjectName(u"resize")
                        self.resizeBtn.setGeometry(QRect(730, 10, 70, 50))
                        self.resizeBtn.setFont(font2)
                        self.resizeBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.resizeBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon6 = QIcon()
                        icon6.addFile(u"imagesForUI\\resize.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.resizeBtn.setIcon(icon6)
                        self.resizeBtn.setIconSize(QSize(30, 30))
                        self.resizeBtn.setCheckable(False)
                        self.resizeBtn.setPopupMode(QToolButton.InstantPopup)
                        self.resizeBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.resizeBtn.setAutoRaise(False)
                        self.selectBtn = QToolButton(self.widgetNavBar)
                        self.selectBtn.setObjectName(u"select")
                        self.selectBtn.setGeometry(QRect(790, 10, 70, 50))
                        self.selectBtn.setFont(font2)
                        self.selectBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.selectBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n")
                        icon7 = QIcon()
                        icon7.addFile(u"imagesForUI\\cursor.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.selectBtn.setIcon(icon7)
                        self.selectBtn.setIconSize(QSize(30, 30))
                        self.selectBtn.setCheckable(False)
                        self.selectBtn.setPopupMode(QToolButton.InstantPopup)
                        self.selectBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.selectBtn.setAutoRaise(False)
                        self.filterBtn = QToolButton(self.widgetNavBar)
                        self.filterBtn.setObjectName(u"filters")
                        self.filterBtn.setGeometry(QRect(850, 10, 70, 50))
                        self.filterBtn.setFont(font2)
                        self.filterBtn.setContextMenuPolicy(Qt.DefaultContextMenu)
                        self.filterBtn.setStyleSheet(u"QToolButton{\n"
                "	background-color:#282e36;\n"
                "	border:none;\n"
                "	color:white;\n"
                "}\n"
                "QToolButton:hover{\n"
                "		background-color:#3e424a;\n"
                "}")
                        icon8 = QIcon()
                        icon8.addFile(u"imagesForUI\\gear.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.filterBtn.setIcon(icon8)
                        self.filterBtn.setIconSize(QSize(30, 30))
                        self.filterBtn.setCheckable(False)
                        self.filterBtn.setPopupMode(QToolButton.InstantPopup)
                        self.filterBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                        self.filterBtn.setAutoRaise(False)
                        self.widgetSidePanel = QWidget(self.centralwidget)
                        self.widgetSidePanel.setObjectName(u"widgetSidePanel")
                        self.widgetSidePanel.setGeometry(QRect(1180, 60, 420, 781))
                        self.widgetSidePanel.setStyleSheet(u"background-color:#36454F;")
                        self.histogrammeLabel = QLabel(self.widgetSidePanel)
                        self.histogrammeLabel.setObjectName(u"histogrammeLabel")
                        font3 = QFont()
                        font3.setFamilies([u"Arial Black"])
                        font3.setPointSize(14)
                        font3.setBold(False)
                        font3.setItalic(False)
                        font3.setStrikeOut(False)
                        font3.setStyleStrategy(QFont.PreferDefault)
                        self.histogrammeLabel.setFont(font3)
                        self.histogrammeLabel.setStyleSheet(u"color:#282d36;")

                        self.histogrammeLabel.setGeometry(QRect(-40, 0, 500, 320))
                        self.histogrammeLabel.setAlignment(Qt.AlignCenter)


                        self.line_3 = QFrame(self.widgetSidePanel)
                        self.line_3.setObjectName(u"line_3")
                        self.line_3.setGeometry(QRect(50, 310, 320, 20))
                        self.line_3.setFrameShape(QFrame.HLine)
                        self.line_3.setFrameShadow(QFrame.Sunken)

                        self.widget_10 = QWidget(self.widgetSidePanel)
                        self.widget_10.setObjectName(u"widget_6")
                        self.widget_10.setGeometry(QRect(0, 450, 421, 131))

                        self.compressBox = QComboBox(self.widgetSidePanel)
                        self.compressBox.setObjectName(u"Compression")
                        self.compressBox.setGeometry(QRect(35,600,350,50))
                        self.compressBox.setStyleSheet(u"background-color:#ffffff;")

                        self.compressButton = QPushButton(self.widgetSidePanel)
                        self.compressButton.setObjectName(u"Compresser")
                        self.compressButton.setGeometry(QRect(160,680,100,50))
                        self.compressButton.setText("Compresser")
                        self.compressButton.setStyleSheet(u"background-color:#ffffff;")

                        self.widgetZoom = QStackedWidget(self.centralwidget)
                        self.widgetZoom.setObjectName(u"widgetZoom")
                        self.widgetZoom.setGeometry(QRect(1000, 60, 180, 50))
                        self.widgetZoom.setStyleSheet(u"background-color:#ffffff;")

                        self.zoomIn = QToolButton(self.widgetZoom)
                        self.zoomIn.setObjectName(u"zoomIn")
                        self.zoomIn.setGeometry(QRect(60, 0, 50, 50))
                        self.zoomIn.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon18 = QIcon()
                        icon18.addFile(u"imagesForUI\\zoom-in.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.zoomIn.setIcon(icon18)
                        self.zoomIn.setIconSize(QSize(32, 32))
                        self.zoomOut = QToolButton(self.widgetZoom)
                        self.zoomOut.setObjectName(u"zoomOut")
                        self.zoomOut.setGeometry(QRect(0, 0, 50, 50))
                        self.zoomOut.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon19 = QIcon()
                        icon19.addFile(u"imagesForUI\\zoom-out.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.zoomOut.setIcon(icon19)
                        self.zoomOut.setIconSize(QSize(32, 32))

                        self.diff = QToolButton(self.widgetZoom)
                        self.diff.setObjectName(u"zoomOut")
                        self.diff.setGeometry(QRect(120, 0, 50, 50))
                        self.diff.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon20 = QIcon()
                        icon20.addFile(u"imagesForUI\\before-after.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.diff.setIcon(icon20)
                        self.diff.setIconSize(QSize(32, 32))




                        self.widgetNavBar2 = QStackedWidget(self.centralwidget)
                        self.widgetNavBar2.setObjectName(u"widget_3")
                        self.widgetNavBar2.setGeometry(QRect(0, 60, 1000, 50))
                        self.widgetNavBar2.setStyleSheet(u"background-color:#ffffff;")
                        self.widgetRotate = QWidget()
                        self.widgetRotate.setObjectName(u"widget_5")
                        self.widgetRotate.setGeometry(QRect(0, 0, 1180, 50))
                        self.widgetRotate.setStyleSheet(u"background-color:;")
                        self.rotationLeft = QToolButton(self.widgetRotate)
                        self.rotationLeft.setObjectName(u"rotationLeft")
                        self.rotationLeft.setGeometry(QRect(10, 0, 50, 50))
                        self.rotationLeft.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon10 = QIcon()
                        icon10.addFile(u"imagesForUI\\rotate-left.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.rotationLeft.setIcon(icon10)
                        self.rotationLeft.setIconSize(QSize(32, 32))
                        self.rotationRight = QToolButton(self.widgetRotate)
                        self.rotationRight.setObjectName(u"rotationRight")
                        self.rotationRight.setGeometry(QRect(80, 0, 50, 50))
                        self.rotationRight.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon11 = QIcon()
                        icon11.addFile(u"imagesForUI\\rotate-right.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.rotationRight.setIcon(icon11)
                        self.rotationRight.setIconSize(QSize(32, 32))
                        self.flipHorizontal = QToolButton(self.widgetRotate)
                        self.flipHorizontal.setObjectName(u"flipHorizontal")
                        self.flipHorizontal.setGeometry(QRect(150, 0, 50, 50))
                        self.flipHorizontal.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon12 = QIcon()
                        icon12.addFile(u"imagesForUI\\flip.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.flipHorizontal.setIcon(icon12)
                        self.flipHorizontal.setIconSize(QSize(32, 32))
                        self.flipVertical = QToolButton(self.widgetRotate)
                        self.flipVertical.setObjectName(u"flipVertical")
                        self.flipVertical.setGeometry(QRect(220, 0, 50, 50))
                        self.flipVertical.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon13 = QIcon()
                        icon13.addFile(u"imagesForUI\\flip2.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.flipVertical.setIcon(icon13)
                        self.flipVertical.setIconSize(QSize(32, 32))
                        self.line_3 = QFrame(self.widgetRotate)
                        self.line_3.setObjectName(u"line_3")
                        self.line_3.setGeometry(QRect(280, 5, 3, 40))
                        self.line_3.setFrameShape(QFrame.VLine)
                        self.line_3.setFrameShadow(QFrame.Sunken)

                        validator = QIntValidator()
                        self.widgetRotate.raise_()
                        self.widgetResize = QWidget(self.widgetNavBar2)
                        self.widgetResize.setObjectName(u"widgetResize")
                        self.widgetResize.setGeometry(QRect(-1, 0, 1111, 51))
                        self.width = QLabel(self.widgetResize)
                        self.width.setObjectName(u"width")
                        self.width.setGeometry(QRect(30, 5, 71, 41))
                        self.width.setFont(font)
                        self.widthTE = QLineEdit(self.widgetResize)
                        self.widthTE.setObjectName(u"widthTE")
                        self.widthTE.setGeometry(QRect(90, 10, 100, 30))
                        self.widthTE.setFont(font)
                        self.widthTE.setStyleSheet(u"text-align:center;")
                        self.widthTE.setValidator(validator)
                        # self.widthTE.setFrameShape(QFrame.NoFrame)
                        # self.widthTE.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        # self.widthTE.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        self.height = QLabel(self.widgetResize)
                        self.height.setObjectName(u"height")
                        self.height.setGeometry(QRect(220, 5, 71, 41))
                        self.height.setFont(font)
                        self.appliquer = QToolButton(self.widgetResize)
                        self.appliquer.setObjectName(u"Redimensionner")
                        self.appliquer.setGeometry(QRect(580, 10, 150, 30))
                        self.appliquer.setFont(font)
                        validator = QIntValidator()
                        self.heightTE = QLineEdit(self.widgetResize)
                        self.heightTE.setObjectName(u"heightTE")
                        self.heightTE.setGeometry(QRect(280, 10, 100, 30))
                        self.heightTE.setFont(font)
                        self.heightTE.setStyleSheet(u"text-align:center;")
                        self.heightTE.setValidator(validator)
                        # # self.heightTE.setFrameShape(QFrame.NoFrame)
                        # self.heightTE.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        # self.heightTE.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        self.comboBox = QComboBox(self.widgetResize)
                        self.comboBox.setObjectName(u"comboBox")
                        self.comboBox.setFont(font)
                        self.comboBox.setGeometry(QRect(410, 10, 150, 30))
                        self.comboBox.setStyleSheet(u"border:1px solid black;")
                        self.comboBox.setEditable(False)
                        self.comboBox.setMaxVisibleItems(2)
                        self.comboBox.setStyleSheet(u"border:none;background-color:white;")
                        self.widgetResize.raise_()
                        self.widgetSelect = QWidget(self.widgetNavBar2)
                        self.widgetSelect.setObjectName(u"widgetSelect")
                        self.widgetSelect.setGeometry(QRect(-1, 0, 1111, 51))
                        self.selectRec = QToolButton(self.widgetSelect)
                        self.selectRec.setObjectName(u"selectRec")
                        self.selectRec.setGeometry(QRect(10, 0, 50, 50))
                        self.selectRec.setStyleSheet(u"QToolButton{\n"
                "	border:none;\n"
                "}\n"
                "QToolButton:hover{\n"
                "	background-color:#babdbb;\n"
                "}")
                        icon16 = QIcon()
                        icon16.addFile(u"imagesForUI\\select1.png", QSize(), QIcon.Normal, QIcon.Off)
                        self.selectRec.setIcon(icon16)
                        self.selectRec.setIconSize(QSize(32, 32))
                        self.widgetSelect.raise_()


                        self.widget_10.raise_()
                        self.line_3.raise_()

                        self.widget_filter = QWidget(self.widgetNavBar2)
                        self.widget_filter.setObjectName(u"widget_filter")
                        self.widget_filter.setGeometry(QRect(0, 0, 1111, 51))
                        
                        self.filter = QLabel(self.widget_filter)
                        self.filter.setObjectName(u"filter")
                        self.filter.setGeometry(QRect(30, 5, 50, 41))
                        self.filter.setFont(font)
                        
                        self.listFonction = QComboBox(self.widget_filter)
                        self.listFonction.setObjectName(u"listFonction")
                        self.listFonction.setGeometry(QRect(80, 5, 380, 40))
                        self.listFonction.setFont(font)
                        self.listFonction.setFocusPolicy(Qt.NoFocus)
                        self.listFonction.setToolTipDuration(-1)
                        self.listFonction.setAutoFillBackground(False)
                        self.listFonction.setStyleSheet("background-color:white;")

                        self.pointsInteret = QComboBox(self.widget_filter)
                        self.pointsInteret.setStyleSheet("background-color:white;")
                        self.pointsInteret.setValidator(QDoubleValidator(0.0,255.0,2))
                        # self.listFonction.currentTextChanged.connect(self.on_combobox_changed)
                        self.pointsInteret.setHidden(True)
                        self.pointsInteret.setGeometry(QRect(600, 5, 100, 40))
                        self.pointsInteret.addItem("")
                        self.pointsInteret.addItem("SIFT")
                        self.pointsInteret.addItem("Coins de Harris")
                        self.pointsInteret.addItem("Coins de Shi-Tomasi")
                        self.pointsInteret.addItem("Hough")

                        self.line_5 = QFrame(self.widget_filter)
                        self.line_5.setObjectName(u"line_3")
                        self.line_5.setGeometry(QRect(470, 5, 3, 40))
                        self.line_5.setFrameShape(QFrame.VLine)
                        self.line_5.setFrameShadow(QFrame.Sunken)

                        self.selectFiltre = QToolButton(self.widget_filter)
                        self.selectFiltre.setGeometry(QRect(700, 5, 100, 40))
                        self.selectFiltre.setText("Appliquer")

                        self.parametre = QLineEdit(self.widget_filter)
                        self.parametre.setStyleSheet("background-color:white;")
                        self.parametre.setValidator(QDoubleValidator(0.0,255.0,2))
                        # self.listFonction.currentTextChanged.connect(self.on_combobox_changed)
                        self.parametre.setReadOnly(True)
                        self.parametre.setGeometry(QRect(490, 5, 100, 40))

                        self.morpho = QComboBox(self.widget_filter)
                        self.morpho.setStyleSheet("background-color:white;")
                        self.morpho.setValidator(QDoubleValidator(0.0,255.0,2))
                        # self.listFonction.currentTextChanged.connect(self.on_combobox_changed)
                        self.morpho.setHidden(True)
                        self.morpho.setGeometry(QRect(600, 5, 100, 40))
                        self.morpho.addItem("")
                        self.morpho.addItem("Rectangle")
                        self.morpho.addItem("Ellipse")
                        self.widget_filter.raise_()
  

                        # self.image = QLabel(parent=self.widget_4)
                        # self.image.setGeometry(QRect(0, 0, 0, 0))
                        # self.image.setObjectName("image")


                        self.widgetImage = QWidget(self.centralwidget)
                        self.widgetImage.setObjectName(u"widgetImage")
                        self.widgetImage.setGeometry(QRect(0, 109, 1180, 727))
                        self.widgetImage.setStyleSheet("background-color:#ffffff;")

                        self.image = QImage()
                        self.original_image = self.image
                        self.zoom_factor = 1

                        self.imageLabel = QLabel()
                        self.imageLabel.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
                        self.imageLabel.setScaledContents(True)
                        self.imageLabel.setGeometry(QRect(0, 0, 600, 400))
                        # self.imageLabel.mousePressEvent = self.getPixel
                        
                        self.imageLabel.setPixmap(QPixmap().fromImage(self.image))
                        self.imageLabel.setAlignment(Qt.AlignCenter)
                        
                        self.imageLabel.resize(self.imageLabel.pixmap().size())

                        self.scroll_area = QScrollArea(self.widgetImage)
                        self.scroll_area.setBackgroundRole(QPalette.Dark)
                        self.scroll_area.setAlignment(Qt.AlignCenter)
                        self.scroll_area.setGeometry(QRect(0, 0, 1180, 727))
                        #self.scroll_area.setWidgetResizable(False)
                        #scroll_area.setMinimumSize(800, 800)
                        
                        self.scroll_area.setWidget(self.imageLabel)
                        #self.scroll_area.setVisible(False)

                        # self.widgetImage.setCentralWidget(self.scroll_area)
                        # self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.imageLabel)
                        
                        

                        

                        MainWindow.setCentralWidget(self.centralwidget)

                        self.retranslateUi(MainWindow)



                        self.newImageBtn.clicked.connect(self.openFileDialog)
                        self.saveImageBtn.clicked.connect(self.saveImage)
                        self.closeBtn.clicked.connect(self.clearImage)
                        self.rotateBtn.clicked.connect(lambda: self.switchWidget('Rotate'))
                        self.selectBtn.clicked.connect(lambda: self.switchWidget('Select'))
                        self.resizeBtn.clicked.connect(lambda: self.switchWidget('Resize'))
                        self.filterBtn.clicked.connect(lambda: self.switchWidget('Fonction'))
                        # # self.diffrence.clicked.connect(self.negative)
                        self.rotationRight.clicked.connect(lambda : self.rotateImage90("cw"))
                        self.rotationLeft.clicked.connect(lambda : self.rotateImage90("ccw"))
                        self.selectFiltre.clicked.connect(self.filtres)
                        self.appliquer.clicked.connect(self.resizeImage)
                        self.selectRec.clicked.connect(self.crop)
                        self.listFonction.currentTextChanged.connect(self.on_combobox_changed)
                        # self.selectFree.clicked.connect(self.cropFreeSelection)
                        self.flipHorizontal.clicked.connect(lambda : self.flipImage("horizontal"))
                        self.flipVertical.clicked.connect(lambda : self.flipImage("vertical"))
                        self.zoomIn.clicked.connect(lambda: self.zoomOnImage(1.25))
                        self.zoomOut.clicked.connect(lambda: self.zoomOnImage(0.8))
                        self.OrigineBtn.clicked.connect(self.revertToOriginal)
                        self.diff.pressed.connect(self.showDiffrence)
                        self.diff.released.connect(self.releaseDiffrence)
                        self.compressButton.clicked.connect(self.compression)


                        self.widgetNavBar2.addWidget(self.widgetRotate)
                        self.widgetNavBar2.addWidget(self.widgetResize)
                        self.widgetNavBar2.addWidget(self.widgetSelect)
                        self.widgetNavBar2.addWidget(self.widget_filter)
                        self.widgetNavBar2.setCurrentWidget(self.widgetRotate)
                        self.widgetNavBar2.setStyleSheet("background-color:#ffffff;");
                        # self.switchWidget('Rotate')

                        QMetaObject.connectSlotsByName(MainWindow)
                # setupUi
        def retranslateUi(self, MainWindow):
                MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Photo editeur", None))
                # self.histogrammeLabel.setText(QCoreApplication.translate("MainWindow", u"Histograme", None))
                # self.fonctionLabel.setText(QCoreApplication.translate("MainWindow", u"Fonctions", None))
                # self.rotationDegree.setText(QCoreApplication.translate("MainWindow", u"180 \u00ba", None))
                self.width.setText(QCoreApplication.translate("MainWindow", u"Largeur :", None))
                self.height.setText(QCoreApplication.translate("MainWindow", u"Hauteur :", None))
                self.filter.setText(QCoreApplication.translate("MainWindow", u"Filtres :", None))
                self.appliquer.setText(QCoreApplication.translate("MainWindow", u"Redimensionner", None))
                self.comboBox.addItem('Pixels')
                self.comboBox.addItem('Poucentage')

                self.compressBox.addItem("Compression par LZW")
                self.compressBox.addItem("Compression par Ondelette")
                self.compressBox.addItem("Compression par Huffman")

                self.listFonction.addItem('')
                self.listFonction.addItem('Image Negative')
                self.listFonction.addItem('Binarisation OTSU')
                self.listFonction.addItem('Binarisation Manuelle')
                self.listFonction.addItem('Egalisation Histogramme')
                self.listFonction.addItem('Etirement Histogramme')
                self.listFonction.addItem('Filtre Gaussien')
                self.listFonction.addItem('Filtre Moyenneur')
                self.listFonction.addItem('Filtre Médiane')
                self.listFonction.addItem('Erosion')
                self.listFonction.addItem('Dilatation')
                self.listFonction.addItem('Ouverture')
                self.listFonction.addItem('Fermeture')
                self.listFonction.addItem('Filtrage Morphologique')
                self.listFonction.addItem('Conteur Gradient')
                self.listFonction.addItem('Conteur Sobel')
                self.listFonction.addItem('Conteur Robert')
                self.listFonction.addItem('Conteur Laplacien')
                self.listFonction.addItem('Ségmentation par KMeans')
                self.listFonction.addItem('Ségmentation par Croissance de régions D')
                self.listFonction.addItem('Ségmentation par Partition de régions D')
                self.listFonction.addItem('Detection des points d\'intérêt')

        def is_grey_scale(self,img):
                w, h,c = img.shape
                for i in range(w):
                        for j in range(h):
                                r, g, b = img[i][j]
                                if r != g != b: 
                                        return False
                return True


        def on_combobox_changed(self, value):   
                print("combobox changed", value)
                if(value == "Binarisation Manuelle" or
                    value == "Filtre Moyenneur" or
                      value == "Filtre Gaussien" or
                      value == "Filtre Médiane" or 
                      value == "Erosion" or
                      value == "Dilatation" or
                      value == "Ouverture" or
                      value == "Fermeture" or 
                        value == "Filtrage Morphologique") :
                        self.parametre.setReadOnly(False)
                        self.pointsInteret.setHidden(True)
                        if (value == "Erosion" or
                                value == "Dilatation" or
                                value == "Ouverture" or
                                value == "Fermeture" or
                                value == "Filtrage Morphologique"):
                                        self.morpho.setHidden(False)
                elif value == "Detection des points d\'intérêt":
                        self.pointsInteret.setHidden(False)
                elif value == "Ségmentation par Croissance de régions D":
                        self.imageLabel.mousePressEvent = self.getPixel
                        self.pointsInteret.setHidden(True)
                else:
                        self.parametre.setReadOnly(True)
                        self.morpho.setHidden(True)
                        self.pointsInteret.setHidden(True)


        def filt_moy(self,I, n):
                
                h = np.ones((n,n))

                h = h / np.sum(h)
                
                h = np.flipud(np.fliplr(h))
                
                I_filt = convolve2d(I, h, mode='same')
                
                return I_filt

        def filtres(self):
                if self.image.isNull() == False:
                        text=self.listFonction.currentText()
                        if text == "Image Negative" : self.negative()
                        elif text == "Binarisation OTSU" : self.binarisationImage()
                        elif text == "Egalisation Histogramme": self.equalizeHistogramme()
                        elif text == "Etirement Histogramme": self.EtirerHistogramme()
                        elif text == "Binarisation Manuelle" : self.binarisationLocal()
                        elif text == "Filtre Moyenneur" : self.filtre_moyenneur()
                        elif text == "Filtre Gaussien" : self.filtreGaussien()
                        elif text == "Filtre Médiane" : self.filtreMediane()
                        elif text == "Erosion" or text == "Dilatation" or text == "Ouverture" or text == "Fermeture" or text == "Filtrage Morphologique":
                                self.Morphologie()
                        elif text == "Conteur Sobel" : self.conteur()
                        elif text == "Conteur Robert" : self.conteur()
                        elif text == "Conteur Laplacien" : self.conteur()
                        elif text == "Conteur Gradient" : self.conteur()
                        elif text == "Ségmentation par KMeans" : self.segmentations()
                        elif text == "Ségmentation par Partition de régions D" : self.segmentationsParPartition()
                        elif text == "Ségmentation par Croissance de régions D" : self.segmentationsCroissance()
                        elif text == "Detection des points d\'intérêt" : self.detectionPointInter()



        def switchWidget(self,button):
                if button == 'Rotate' :self.widgetNavBar2.setCurrentWidget(self.widgetRotate)
                if button == 'Resize' :self.widgetNavBar2.setCurrentWidget(self.widgetResize)
                if button == 'Select' :self.widgetNavBar2.setCurrentWidget(self.widgetSelect)
                if button == 'Fonction' :self.widgetNavBar2.setCurrentWidget(self.widget_filter)
        def openFileDialog(self):
                file_types = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.gif)"
                fileDialog = QFileDialog()
                self.fileName = fileDialog.getOpenFileName(None, "Open File", "", file_types)
                if(self.fileName[0] != ''):
                        self.zoom_factor = 1
                        self.image = QImage(self.fileName[0])
                        self.original_image = self.image.copy()
                        w = self.image.width()
                        h = self.image.height()
                        if w > 1080 or h > 900:
                                self.imageLabel.setPixmap(QPixmap().fromImage(self.image).scaled(900,500,aspectRatioMode=Qt.KeepAspectRatio));
                        else:
                                self.imageLabel.setPixmap(QPixmap().fromImage(self.image))
                        self.imageLabel.resize(self.imageLabel.pixmap().size())
                        self.histogramImage()
                else:pass
        def zoomOnImage(self, zoom_value):
                """Zoom in and zoom out."""
                
                self.zoom_factor *= zoom_value
                self.imageLabel.resize(self.zoom_factor * self.imageLabel.pixmap().size())

                self.adjustScrollBar(self.scroll_area.horizontalScrollBar(), zoom_value)
                self.adjustScrollBar(self.scroll_area.verticalScrollBar(), zoom_value)

                self.OrigineBtn.setEnabled(self.zoom_factor < 4.0)
                self.closeBtn.setEnabled(self.zoom_factor > 0.333)
        def adjustScrollBar(self, scroll_bar, value):
                """Adjust the scrollbar when zooming in or out."""
                scroll_bar.setValue(int(int(value * scroll_bar.value()) + ((value - 1) * scroll_bar.pageStep()/2)))

        def saveImage(self):
                """Save the image displayed in the label."""
                #TODO: Add different functionality for the way in which the user can save their image.
                if self.image.isNull() == False:
                        image_file, _ = QFileDialog.getSaveFileName(self.imageLabel, "Save Image", 
                                "", "PNG Files (*.png);;JPG Files (*.jpeg *.jpg );;Bitmap Files (*.bmp);;\
                                GIF Files (*.gif)")

                        if image_file and self.image.isNull() == False:
                                self.image.save(image_file)
                        else:
                                QMessageBox.information(self.imageLabel, "Error", 
                                "Unable to save image.", QMessageBox.Ok)
                else:
                        QMessageBox.information(self, "Empty Image", 
                        "There is no image to save.", QMessageBox.Ok)

        def clearImage(self):
                self.image = QImage()    
                self.imageLabel.setPixmap(QPixmap().fromImage(self.image))
                self.histogrammeLabel.clear()
                self.imageLabel.resize(self.imageLabel.pixmap().size())   

        def revertToOriginal(self):
                self.image = self.original_image
                self.imageLabel.setPixmap(QPixmap().fromImage(self.image))
                self.histogramImage()
                self.imageLabel.repaint()

        def showDiffrence(self):
                self.image = QImage(self.imageLabel.pixmap().copy())
                self.imageLabel.setPixmap(QPixmap().fromImage(self.original_image))
                self.histogramImage()
                self.imageLabel.repaint()

        def releaseDiffrence(self):
                self.imageLabel.setPixmap(QPixmap().fromImage(self.image))
                self.histogramImage()
                self.imageLabel.repaint()
                


        def rotateImage90(self, direction):
                """Rotate image 90º clockwise or counterclockwise."""
                if self.image.isNull() == False:
                        if direction == "cw":
                                transform90 = QTransform().rotate(90)
                        elif direction == "ccw":
                                transform90 = QTransform().rotate(-90)

                        pixmap = QPixmap(self.image)

                        #TODO: Try flipping the height/width when flipping the image

                        rotated = pixmap.transformed(transform90, mode=Qt.SmoothTransformation)
                        self.imageLabel.resize(self.image.height(), self.image.width())
                        #rotated = pixmap.trueMatrix(transform90, pixmap.width, pixmap.height)
                        
                        #self.image_label.setPixmap(rotated.scaled(self.image_label.size(), 
                        #    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        self.image = QImage(rotated) 
                        #self.setPixmap(rotated)
                        self.imageLabel.setPixmap(rotated.scaled(self.imageLabel.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
                        self.imageLabel.repaint() # repaint the child widget
                else:
                        # No image to rotate
                        pass
        def flipImage(self, axis):
                if self.image.isNull() == False:
                        if axis == "horizontal":
                                flip_h = QTransform().scale(-1, 1)
                                pixmap = QPixmap(self.image)
                                flipped = pixmap.transformed(flip_h)
                        elif axis == "vertical":
                                flip_v = QTransform().scale(1, -1)
                                pixmap = QPixmap(self.image)
                                flipped = pixmap.transformed(flip_v)
                        self.image = QImage(flipped)
                        self.imageLabel.setPixmap(flipped)
                        self.imageLabel.repaint()
                else:
                        # No image to flip
                        pass
        def resizeImage(self):
                """Resize image."""
                #TODO: Resize image by specified size
                if self.image.isNull() == False:
                        img_x,img_y = self.image.width(),self.image.height()
                        print(img_x,img_y)

                        if self.heightTE.text() != "" and self.widthTE.text() != "" :
                                x=int(self.widthTE.text())
                                y=int(self.heightTE.text())
                                if self.comboBox.currentText() == "Pixels":
                                        x,y = x/img_x,y/img_y
                                else:
                                        x,y = x/100,y/100
                        else:
                                x,y = 0.5,0.5
                        print(x,y)
                        resize = QTransform().scale(x, y)
                                

                        pixmap = QPixmap(self.image)


                        resized_image = pixmap.transformed(resize, mode=Qt.SmoothTransformation)
                        self.image = QImage(resized_image) 
                        self.imageLabel.setPixmap(resized_image)
                        self.imageLabel.resize(self.imageLabel.pixmap().size())
                        
                        self.imageLabel.setScaledContents(True)
                        self.imageLabel.repaint() 
                else:
                # No image to rotate
                        pass
        def convertQImageToMat(self,incomingImage):
                incomingImage = incomingImage.convertToFormat(QImage.Format_RGBX8888)
                ptr = incomingImage.constBits()
                ptr.setsize(incomingImage.byteCount())
                cv_im_in = np.array(ptr, copy=True).reshape(incomingImage.height(), incomingImage.width(), 4)
                cv_im_in = cv2.cvtColor(cv_im_in, cv2.COLOR_BGRA2RGB)
                return cv_im_in
        
        def negative(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                
                x,y,c = image.shape
                if self.is_grey_scale(image):
                        bytes_per_line = 1 * x
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        for i in range(x):
                                for j in range(y):
                                        image[i][j]=255-image[i][j]
                        self.image = image
                        q_image = QImage(image.data, y, x, bytes_per_line, QImage.Format.Format_Grayscale8)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                else:
                        bytes_per_line = 3 * y
                        for i in range(x):
                                for j in range(y):
                                        for k in range(3):
                                                image[i][j][k]=255-image[i][j][k]
                        cv2.imwrite('figure/negative.jpg',image) # Save the negative image
                        imageNegative = cv2.imread('figure/negative.jpg') # Retrieve it so we can display it
                        try:  
                                os.remove('figure/negative.jpg') # Delete the negative image from folder
                        except:pass
                        q_image = QImage(imageNegative.data, y, x, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def binarisationImage(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        th, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=image[:,:,0]
                        G=image[:,:,1]
                        B=image[:,:,2]
                        _, R_image = cv2.threshold(R, 0, 255, cv2.THRESH_OTSU)
                        _, G_image = cv2.threshold(G, 0, 255, cv2.THRESH_OTSU)
                        _, B_image = cv2.threshold(B, 0, 255, cv2.THRESH_OTSU)
                        image = np.dstack((R_image,G_image,B_image))
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def binarisationLocal(self):
                thres = self.parametre.text()
                if(thres == ""):
                        thres = 128
                else:
                        thres=int(thres)
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        th, image = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        print(th)
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=image[:,:,0]
                        G=image[:,:,1]
                        B=image[:,:,2]
                        _, R_image = cv2.threshold(R, thres, 255, cv2.THRESH_BINARY)
                        _, G_image = cv2.threshold(G, thres, 255, cv2.THRESH_BINARY)
                        _, B_image = cv2.threshold(B, thres, 255, cv2.THRESH_BINARY)
                        image = np.dstack((R_image,G_image,B_image))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def histogramImage(self):
                picVal = self.imageLabel.pixmap()
                image = picVal.toImage()
                image=self.convertQImageToMat(image)
                try:    
                        self.histogrammeLabel.clear()
                        os.remove('figure\\fig.png')
                except:pass
                
                if self.is_grey_scale(image):
                        editedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([editedImage],[0],None,[256],[0,256])
                        plt.clf()
                        plt.plot(hist)
                        print('Lets edite')
                        plt.savefig('figure\\fig.png')
                        fig=cv2.imread('figure\\fig.png')
                else:
                        B=image[:,:,0]
                        G=image[:,:,1]
                        R=image[:,:,2]
                        histB=cv2.calcHist([B],[0],None,[256],[0,256])
                        histG=cv2.calcHist([G],[0],None,[256],[0,256])
                        histR=cv2.calcHist([R],[0],None,[256],[0,256])
                        plt.clf()
                        plt.plot(histB,color='blue')
                        plt.plot(histG,color='green')
                        plt.plot(histR,color='red')
                        plt.savefig('figure\\fig.png')
                        fig=cv2.imread('figure\\fig.png')
                
                finalImage = cv2.resize(fig,(0,0),fx=0.6,fy=0.6)
                
                binary_image = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
                height, width, channel = binary_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(binary_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                # img = QPixmap(fileName[0])
                self.histogrammeLabel.setPixmap(QPixmap.fromImage(q_image))

        def EtirerHistogramme(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        maximum = image.max()
                        minimum = image.min()
                        x=image.shape[1]
                        y=image.shape[0]
                        for i in range(x):
                                for j in range(y):
                                        image[i][j] = (255 /(maximum-minimum))*(image[i][j]-minimum)

                        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        bytes_per_line = 3 * x
                        q_image = QImage(image.data, x, y, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        
                else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=image[:,:,0]
                        G=image[:,:,1]
                        B=image[:,:,2] 
                        
                        maximumR = R.max()
                        minimumR = R.min()
                        x=R.shape[1]
                        y=R.shape[0]

                        maximumG = G.max()
                        minimumG = G.min()

                        maximumB = B.max()
                        minimumB = B.min()
                        for i in range(y):
                                for j in range(x):
                                        R[i][j] = (255 /(maximumR-minimumR))*(R[i][j]-minimumR)
                                        G[i][j] = (255 /(maximumG-minimumG))*(G[i][j]-minimumG)
                                        B[i][j] = (255 /(maximumB-minimumB))*(B[i][j]-minimumB)
                        
                        image= np.dstack((R,G,B))
                        bytes_per_line = 3 * x
                        q_image = QImage(image.data, x, y, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()


        def equalizeHistogramme(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        image= cv2.equalizeHist(image)
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()
                else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=image[:,:,0]
                        G=image[:,:,1]
                        B=image[:,:,2]
                        R_image = cv2.equalizeHist(R)
                        G_image = cv2.equalizeHist(G)
                        B_image = cv2.equalizeHist(B)
                        image = np.dstack((R_image,G_image,B_image))
                        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()

        
        def crop(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                imageS=self.convertQImageToMat(pic)
                imageS = cv2.cvtColor(imageS,cv2.COLOR_BGR2RGB)
                image=imageS
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                x, y, w, h = cv2.selectROI('image', image, False)

                # Draw the rectangle on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Display the image with the rectangle
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                imageS=cv2.cvtColor(imageS,cv2.COLOR_BGR2RGB)
                imageFin = imageS[y:y+h, x:x+w]
                imageFin = cv2.cvtColor(imageFin,cv2.COLOR_BGR2RGB)
                height, width, channel = imageFin.shape
                bytes_per_line = 3 * width
                q_image = QImage(imageFin.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))


        def filtre_moyenneur(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                val = self.parametre.text()
                if val == "":
                        taille = 3
                else:
                        taille = int(val)
                if self.is_grey_scale(image):
                        images = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        imageRes = self.filt_moy(images,taille)
                        imageRes = imageRes.astype(np.uint8)
                        imageRes = cv2.cvtColor(imageRes,cv2.COLOR_BGR2RGB)
                        height, width, channel = imageRes.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()

                else:
                        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=images[:,:,0]
                        G=images[:,:,1]
                        B=images[:,:,2]
                        R_image = self.filt_moy(R,taille)
                        G_image = self.filt_moy(G,taille)
                        B_image = self.filt_moy(B,taille)
                        image = np.dstack((R_image,G_image,B_image))
                        imageResultat = image.astype(np.uint8)
                        height, width, channel = imageResultat.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageResultat.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()

        def filtreGaussien(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                val = self.parametre.text()
                if val == "":
                        sigma = 1.5
                else:
                        sigma = float(val)
                
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        image_filtered = gaussian_filter(image, sigma)
                        image_filtered = image_filtered.astype(np.uint8)
                        imageRes = cv2.cvtColor(image_filtered,cv2.COLOR_BGR2RGB)
                        height, width, channel = imageRes.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()
                else:
                        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=images[:,:,0]
                        G=images[:,:,1]
                        B=images[:,:,2]
                        R_image = gaussian_filter(R, sigma)
                        G_image = gaussian_filter(G, sigma)
                        B_image = gaussian_filter(B, sigma)
                        image = np.dstack((R_image,G_image,B_image))
                        imageResultat = image.astype(np.uint8)
                        height, width, channel = imageResultat.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageResultat.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()

        def filtreMediane(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                val = self.parametre.text()
                if val == "":
                        size = 3
                else:
                        size = int(val)
                
                if self.is_grey_scale(image):
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        image_filtered = median_filter(image, size)
                        image_filtered = image_filtered.astype(np.uint8)
                        imageRes = cv2.cvtColor(image_filtered,cv2.COLOR_BGR2RGB)
                        height, width, channel = imageRes.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()
                else:
                        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        R=images[:,:,0]
                        G=images[:,:,1]
                        B=images[:,:,2]
                        R_image = median_filter(R, size)
                        G_image = median_filter(G, size)
                        B_image = median_filter(B, size)
                        image = np.dstack((R_image,G_image,B_image))
                        imageResultat = image.astype(np.uint8)
                        height, width, channel = imageResultat.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(imageResultat.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.image = q_image
                        self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                        self.histogramImage()

        def Morphologie(self):
                size = self.parametre.text()
                if size != "":
                        size = size.split(",")
                        x = int(size[0])
                        y = int(size[1])
                else:
                        x = y = 3
                kernelType = self.morpho.currentText()
                if kernelType == "Rectangle":
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(x,y))
                else:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(x,y))

                morphologie = self.listFonction.currentText() 
                
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                if morphologie == "Erosion":
                        imageResultant =cv2.erode(image,kernel)
                elif morphologie =="Dilatation":
                        imageResultant =cv2.dilate(image,kernel) 
                elif morphologie == "Ouverture":
                        imageResultant = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
                elif morphologie == "Fermeture":
                        imageResultant = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                elif morphologie == "Filtrage Morphologique":
                        imageResultant = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
                        imageResultant = cv2.morphologyEx(imageResultant, cv2.MORPH_CLOSE, kernel)

                
                imageRes = cv2.cvtColor(imageResultant,cv2.COLOR_BGR2RGB)
                height, width, channel = imageRes.shape
                bytes_per_line = 3 * width
                q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def getPixel(self, event):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                x = event.pos().x()
                y = event.pos().y()
                x = int(x * image.shape[1] / self.imageLabel.width())
                y = int(y * image.shape[0] / self.imageLabel.height())

                self.parametre.setText(str(x)+","+str(y))

        def segmentationsCroissance(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                img = image.copy()
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                height, width = image.shape[:2]
                out_img = np.zeros((height, width), dtype=np.uint8)
                fil = []

                seed = self.parametre.text()
                if seed != "":
                        seed = seed.split(",")
                        x = int(seed[0])
                        y = int(seed[1])
                        seed = (x,y)
                else:
                        seed = (10,10)

                threshold = 100
                # Ajout du point de départ à la file
                fil.append(seed)

                # Boucle principale de croissance de région
                while len(fil) > 0:
                        # Obtention du point suivant dans la file
                        current_point = fil.pop(0)

                        # Obtention des coordonnées du point
                        x, y = current_point

                        # Vérification de la validité des coordonnées
                        if x < 0 or y < 0 or x >= height or y >= width:
                                continue

                        # Vérification si le pixel a déjà été visité
                        if out_img[x][y] > 0:
                                continue

                        # Vérification de la différence de valeur de pixel
                        if abs(int(image[x][y]) - int(image[seed])) > threshold:
                                continue

                        # Ajout du pixel à la région
                        out_img[x][y] = 255

                        fil.append((x - 1, y))
                        fil.append((x + 1, y))
                        fil.append((x, y - 1))
                        fil.append((x, y + 1))
                inverted = cv2.bitwise_not(out_img)
                result = cv2.bitwise_and(img,img, mask= inverted)
                out_img = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
                
                height, width,channels = out_img.shape
                bytes_per_line = 3 * width
                q_image = QImage(out_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()
        
        def segmentations(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

                pixel_values = image.reshape(-1,3)

                kmeans = KMeans(n_clusters=2)
                kmeans.fit(pixel_values)

                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                segmented_image = centers[labels].reshape(image.shape)

                segmented_image = segmented_image.astype(np.uint8)
                height, width, channel = segmented_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(segmented_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def segmentationsParPartition(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                # Segmenter l'image en utilisant l'algorithme SLIC
                segments = slic(image, n_segments=100, compactness=10)

                # Colorier les régions segmentées
                segmented_image = label2rgb(segments, image, kind='avg')

                height, width, channel = segmented_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(segmented_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def seuil(self,image):
                seuil = image.mean()
                prevSeuil = 0
                while(prevSeuil != seuil):
                        prevSeuil = seuil
                        arr1 = image[image < seuil]
                        arr2 = image[image >=seuil]
                        min1 = arr1.mean()
                        min2 = arr2.mean()
                        seuil = (min1+min2)/2

                return seuil

        def detectionPointInter(self):
                pointsInteretAlgo = self.pointsInteret.currentText() 
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                if pointsInteretAlgo == "Coins de Harris":
                        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                        dst = cv2.dilate(dst, None)
                        image[dst > 0.01 * dst.max()] = [0, 0, 255]
                elif pointsInteretAlgo == "Coins de Shi-Tomasi":
                        # Détecter les coins de Shi-Tomasi
                        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
                        corners = np.int0(corners)
                        for i in corners:
                                x, y = i.ravel()
                                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                elif pointsInteretAlgo == "SIFT":
                        sift = cv2.SIFT_create()

                        keypoints = sift.detect(image, None)

                        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                else:
                        surf = cv2.ORB_create()
                        keypoints, descriptors = surf.detectAndCompute(gray, None)
                        image = cv2.drawKeypoints(image, keypoints, None, (0, 255, 0), 4)

                imageRes = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                height, width, channel = imageRes.shape
                bytes_per_line = 3 * width
                q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()

        def encodeHuffman(self, image):
                freq = defaultdict(int)
                for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                                freq[image[i,j]] += 1
                heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
                heapq.heapify(heap)
                while len(heap) > 1:
                        lo = heapq.heappop(heap)
                        hi = heapq.heappop(heap)
                        for pair in lo[1:]:
                                pair[1] = '0' + pair[1]
                        for pair in hi[1:]:
                                pair[1] = '1' + pair[1]
                        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
                code = dict(heapq.heappop(heap)[1:])
                encoded_image = []
                for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                                encoded_image.append(code[image[i,j]])
                return encoded_image, code
        
        def decodeHuffman(self, encoded_image, code, shape):
                inv_code = {v: k for k, v in code.items()}
                decoded_image = []
                current_code = ''
                for code in encoded_image:
                        current_code += code
                        if current_code in inv_code:
                                decoded_image.append(inv_code[current_code])
                                current_code = ''
                decoded_image = np.array(decoded_image).reshape(shape)
                return decoded_image

        def compression(self):
                from LZW import LZW
                import os
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)
                compression = self.compressBox.currentText()
                if compression == "Compression par Ondelette":
                        coeffs = pywt.dwt2(image, 'haar')
                        LL, (LH, HL, HH) = coeffs
                        threshold = 30
                        LH[np.abs(LH) < threshold] = 0
                        HL[np.abs(HL) < threshold] = 0
                        HH[np.abs(HH) < threshold] = 0
                        reconstructed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
                        reconstructed = np.uint8(reconstructed)
                        imageRes = cv2.cvtColor(reconstructed,cv2.COLOR_BGR2RGB)
                        
                elif compression == "Compression par LZW":
                        compressor = LZW(image)
                        compressor.compress()
                        decompress = LZW(os.path.join("CompressedFiles",image+"Compressed.lzw"))
                        decompress.decompress()
                        imageCompressed = cv2.imread(os.path.join("DecompressedFiles",image+"Decompressed.tif"))
                        imageRes = cv2.cvtColor(imageCompressed,cv2.COLOR_BGR2RGB)
                else:
                        if len(image.shape) == 3:
                                b,g,r = cv2.split(image)       
                                encoded_b, code = self.encodeHuffman(b)
                                decoded_b = self.decodeHuffman(encoded_b, code, b.shape)
                                encoded_g, code = self.encodeHuffman(g)
                                decoded_g = self.decodeHuffman(encoded_g, code, g.shape)
                                encoded_r, code = self.encodeHuffman(r)
                                decoded_r = self.decodeHuffman(encoded_r, code, r.shape)

                                decoded_image = cv2.merge((decoded_b, decoded_g, decoded_r))
                        else:
                                encoded_image, code = self.encodeHuffman(b)
                                decoded_image = self.decodeHuffman(encoded_image, code, img.shape)

                        imageRes = cv2.cvtColor(decoded_image,cv2.COLOR_BGR2RGB)
                
                height, width, channel = imageRes.shape
                bytes_per_line = 3 * width
                q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.saveImage()
                self.histogramImage()

        def conteur(self):
                picVal = self.imageLabel.pixmap()
                pic = picVal.toImage()
                image=self.convertQImageToMat(pic)                
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                conteur = self.listFonction.currentText() 

                if conteur == "Conteur Sobel":
                        imagex = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
                        imgx = cv2.convertScaleAbs(imagex)
                        imagey = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)
                        imgy = cv2.convertScaleAbs(imagey)
                        imageResultant = cv2.addWeighted(imgx,0.5,imgy,0.5,0)
                elif conteur == "Conteur Robert":
                        robert_x = cv2.filter2D(image,-1, np.array([[-1, 0], [0, 1]], dtype=np.float32))
                        robert_y = cv2.filter2D(image,-1, np.array([[0, -1], [1, 0]], dtype=np.float32))
                        imgx = cv2.convertScaleAbs(robert_x)
                        imgy = cv2.convertScaleAbs(robert_y)
                        imageResultant = cv2.addWeighted(imgx,0.5,imgy,0.5,0)
                elif conteur == "Conteur Laplacien":
                        imageResultant = cv2.Laplacian(image, cv2.CV_8U)
                elif conteur =="Conteur Gradient":
                        kernelSize=3
                        threshold= 100
                        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, kernelSize)
                        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, kernelSize)
                        mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
                        
                        thresholded = cv2.threshold(mag, 126, 255, cv2.THRESH_BINARY)[1]
                        imageResultant = cv2.cvtColor(cv2.convertScaleAbs(thresholded),cv2.COLOR_BGR2RGB)

                imageRes = cv2.cvtColor(imageResultant,cv2.COLOR_BGR2RGB)
                height, width, channel = imageRes.shape
                bytes_per_line = 3 * width
                q_image = QImage(imageRes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.image = q_image
                self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
                self.histogramImage()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = User_Interface()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
