# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Lab417\xio-intrusion-detection\gui\statistics_widget.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_StatisticsWindow(object):
    def setupUi(self, StatisticsWindow):
        StatisticsWindow.setObjectName("StatisticsWindow")
        StatisticsWindow.resize(1080, 720)
        StatisticsWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(StatisticsWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.titleLabel.sizePolicy().hasHeightForWidth())
        self.titleLabel.setSizePolicy(sizePolicy)
        self.titleLabel.setMinimumSize(QtCore.QSize(0, 0))
        self.titleLabel.setMaximumSize(QtCore.QSize(16777215, 60))
        self.titleLabel.setStyleSheet("background-color: rgb(39, 39, 39);\n"
"font: 29pt \"宋体\";\n"
"color: rgb(90, 174, 242);\n"
"color: rgb(255, 255, 255);\n"
"color: rgb(85, 255, 255);")
        self.titleLabel.setScaledContents(True)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setWordWrap(False)
        self.titleLabel.setIndent(0)
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayout.addWidget(self.titleLabel)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1080, 615))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout.setContentsMargins(0, 9, 0, 9)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(0, -1, 8, -1)
        self.horizontalLayout_2.setSpacing(8)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMinimumSize(QtCore.QSize(0, 25))
        self.label_9.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 14pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.label_9.setObjectName("label_9")
        self.verticalLayout_2.addWidget(self.label_9)
        self.productionLineComboBox = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.productionLineComboBox.setMinimumSize(QtCore.QSize(0, 30))
        self.productionLineComboBox.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 11pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.productionLineComboBox.setObjectName("productionLineComboBox")
        self.productionLineComboBox.addItem("")
        self.productionLineComboBox.addItem("")
        self.productionLineComboBox.addItem("")
        self.productionLineComboBox.addItem("")
        self.productionLineComboBox.addItem("")
        self.verticalLayout_2.addWidget(self.productionLineComboBox)
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setMinimumSize(QtCore.QSize(0, 50))
        self.label_8.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(0, 25))
        self.label.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 14pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.startDateTime = QtWidgets.QDateTimeEdit(self.scrollAreaWidgetContents)
        self.startDateTime.setMinimumSize(QtCore.QSize(0, 30))
        self.startDateTime.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 12pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.startDateTime.setObjectName("startDateTime")
        self.verticalLayout_2.addWidget(self.startDateTime)
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QtCore.QSize(0, 50))
        self.label_5.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(0, 25))
        self.label_3.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 14pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.endDateTime = QtWidgets.QDateTimeEdit(self.scrollAreaWidgetContents)
        self.endDateTime.setMinimumSize(QtCore.QSize(0, 30))
        self.endDateTime.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 12pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.endDateTime.setObjectName("endDateTime")
        self.verticalLayout_2.addWidget(self.endDateTime)
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(0, 50))
        self.label_6.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(0, 25))
        self.label_4.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 14pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.timeIntervalComboBox = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.timeIntervalComboBox.setMinimumSize(QtCore.QSize(0, 30))
        self.timeIntervalComboBox.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"font: 11pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.timeIntervalComboBox.setObjectName("timeIntervalComboBox")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.verticalLayout_2.addWidget(self.timeIntervalComboBox)
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMinimumSize(QtCore.QSize(0, 50))
        self.label_7.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.pushButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"font: 14pt \"宋体\";\n"
"color: rgb(85, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_2.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.setStretch(4, 1)
        self.verticalLayout_2.setStretch(6, 1)
        self.verticalLayout_2.setStretch(7, 1)
        self.verticalLayout_2.setStretch(9, 1)
        self.verticalLayout_2.setStretch(10, 1)
        self.verticalLayout_2.setStretch(12, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.graphLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.graphLabel.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.graphLabel.setText("")
        self.graphLabel.setObjectName("graphLabel")
        self.horizontalLayout_2.addWidget(self.graphLabel)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 5)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        StatisticsWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(StatisticsWindow)
        self.statusbar.setStyleSheet("background-color: rgb(29, 29, 29);\n"
"color: rgb(255, 255, 255);")
        self.statusbar.setObjectName("statusbar")
        StatisticsWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(StatisticsWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1080, 23))
        self.menuBar.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"selection-background-color: rgb(100, 100, 100);\n"
"color: rgb(255, 255, 255);")
        self.menuBar.setObjectName("menuBar")
        self.processMenu = QtWidgets.QMenu(self.menuBar)
        self.processMenu.setObjectName("processMenu")
        self.setupMenu = QtWidgets.QMenu(self.menuBar)
        self.setupMenu.setObjectName("setupMenu")
        self.viewMenu = QtWidgets.QMenu(self.menuBar)
        self.viewMenu.setObjectName("viewMenu")
        StatisticsWindow.setMenuBar(self.menuBar)
        self.openConfigFile = QtWidgets.QAction(StatisticsWindow)
        self.openConfigFile.setObjectName("openConfigFile")
        self.start = QtWidgets.QAction(StatisticsWindow)
        self.start.setObjectName("start")
        self.stop = QtWidgets.QAction(StatisticsWindow)
        self.stop.setObjectName("stop")
        self.chen1 = QtWidgets.QAction(StatisticsWindow)
        self.chen1.setObjectName("chen1")
        self.chen2 = QtWidgets.QAction(StatisticsWindow)
        self.chen2.setObjectName("chen2")
        self.li = QtWidgets.QAction(StatisticsWindow)
        self.li.setObjectName("li")
        self.yv1 = QtWidgets.QAction(StatisticsWindow)
        self.yv1.setObjectName("yv1")
        self.yv2 = QtWidgets.QAction(StatisticsWindow)
        self.yv2.setObjectName("yv2")
        self.wang = QtWidgets.QAction(StatisticsWindow)
        self.wang.setObjectName("wang")
        self.pan = QtWidgets.QAction(StatisticsWindow)
        self.pan.setObjectName("pan")
        self.yue = QtWidgets.QAction(StatisticsWindow)
        self.yue.setObjectName("yue")
        self.fullScreen = QtWidgets.QAction(StatisticsWindow)
        self.fullScreen.setObjectName("fullScreen")
        self.exitFullScreen = QtWidgets.QAction(StatisticsWindow)
        self.exitFullScreen.setObjectName("exitFullScreen")
        self.processMenu.addAction(self.start)
        self.processMenu.addAction(self.stop)
        self.setupMenu.addAction(self.openConfigFile)
        self.viewMenu.addAction(self.fullScreen)
        self.viewMenu.addAction(self.exitFullScreen)
        self.menuBar.addAction(self.processMenu.menuAction())
        self.menuBar.addAction(self.viewMenu.menuAction())
        self.menuBar.addAction(self.setupMenu.menuAction())

        self.retranslateUi(StatisticsWindow)
        QtCore.QMetaObject.connectSlotsByName(StatisticsWindow)

    def retranslateUi(self, StatisticsWindow):
        _translate = QtCore.QCoreApplication.translate
        StatisticsWindow.setWindowTitle(_translate("StatisticsWindow", "异常情况统计与可视化"))
        self.titleLabel.setText(_translate("StatisticsWindow", "异常情况统计与可视化"))
        self.label_9.setText(_translate("StatisticsWindow", "生产线："))
        self.productionLineComboBox.setItemText(0, _translate("StatisticsWindow", "sawanini_1"))
        self.productionLineComboBox.setItemText(1, _translate("StatisticsWindow", "sawanini_2"))
        self.productionLineComboBox.setItemText(2, _translate("StatisticsWindow", "zhuanjixia"))
        self.productionLineComboBox.setItemText(3, _translate("StatisticsWindow", "penfenshang"))
        self.productionLineComboBox.setItemText(4, _translate("StatisticsWindow", "baobantongyong"))
        self.label.setText(_translate("StatisticsWindow", "开始时间："))
        self.label_3.setText(_translate("StatisticsWindow", "截止时间："))
        self.label_4.setText(_translate("StatisticsWindow", "分时统计："))
        self.timeIntervalComboBox.setItemText(0, _translate("StatisticsWindow", "一小时以内"))
        self.timeIntervalComboBox.setItemText(1, _translate("StatisticsWindow", "半天以内"))
        self.timeIntervalComboBox.setItemText(2, _translate("StatisticsWindow", "一天以内"))
        self.timeIntervalComboBox.setItemText(3, _translate("StatisticsWindow", "一周以内"))
        self.timeIntervalComboBox.setItemText(4, _translate("StatisticsWindow", "半个月以内"))
        self.timeIntervalComboBox.setItemText(5, _translate("StatisticsWindow", "一个月以内"))
        self.pushButton.setText(_translate("StatisticsWindow", "确定"))
        self.processMenu.setTitle(_translate("StatisticsWindow", "程序"))
        self.setupMenu.setTitle(_translate("StatisticsWindow", "设置"))
        self.viewMenu.setTitle(_translate("StatisticsWindow", "显示"))
        self.openConfigFile.setText(_translate("StatisticsWindow", "打开配置文件"))
        self.start.setText(_translate("StatisticsWindow", "启动"))
        self.stop.setText(_translate("StatisticsWindow", "终止"))
        self.chen1.setText(_translate("StatisticsWindow", "智能异常事件监测与保护系统"))
        self.chen2.setText(_translate("StatisticsWindow", "产品工件智能角度检测系统"))
        self.li.setText(_translate("StatisticsWindow", "传感器数据采集分析与可视化系统"))
        self.yv1.setText(_translate("StatisticsWindow", "生产线报警智能检测分析系统"))
        self.yv2.setText(_translate("StatisticsWindow", "工厂CPS产线建模与分析系统"))
        self.wang.setText(_translate("StatisticsWindow", "工厂侧板效率智能检测系统"))
        self.pan.setText(_translate("StatisticsWindow", "产品包装配件完整性监测系统"))
        self.yue.setText(_translate("StatisticsWindow", "厚板线OEE效率检测系统"))
        self.fullScreen.setText(_translate("StatisticsWindow", "全屏模式"))
        self.exitFullScreen.setText(_translate("StatisticsWindow", "退出全屏"))
