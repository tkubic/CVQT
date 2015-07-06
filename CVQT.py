import sys
from PyQt4 import QtGui, QtCore
from CVQTui import Ui_MainWindow
import cv2
import numpy as np

class Main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupSignals()

    def setupSignals(self):
        self.ui.btnReset.clicked.connect(self.reset)
        self.ui.btnApply.clicked.connect(self.updateWorkingImg)
        self.ui.rbOrigImg.toggled.connect(self.switchMainImage)
        self.ui.actionOpen.triggered.connect(self.fileOpen)

        self.ui.listGreyscale.currentItemChanged.connect(self.toGreyscale)
        self.ui.sbLH.valueChanged.connect(self.hsvMask)
        self.ui.sbLS.valueChanged.connect(self.hsvMask)
        self.ui.sbLV.valueChanged.connect(self.hsvMask)
        self.ui.sbUH.valueChanged.connect(self.hsvMask)
        self.ui.sbUS.valueChanged.connect(self.hsvMask)
        self.ui.sbUV.valueChanged.connect(self.hsvMask)

        self.ui.btnCannyEdges.clicked.connect(self.auto_canny)
        self.ui.btnEqualizeHist.clicked.connect(self.equalizeHist)
        self.ui.btnCLAHE.clicked.connect(self.clahe)
        self.ui.btnBlurAveraging.clicked.connect(self.blurAveraging)
        self.ui.btnBlurGaussian.clicked.connect(self.blurGaussian)
        self.ui.btnBlurMedian.clicked.connect(self.blurMedian)
        self.ui.btnFilterBilateral.clicked.connect(self.filterBilateral)
        # self.ui.cbThreshType.currentIndexChanged.connect(self.getThreshType)
        self.ui.btnThreshGlobal.clicked.connect(self.threshGlobal)
        self.ui.btnThreshGaussian.clicked.connect(self.threshGaussian)
        self.ui.btnThreshMean.clicked.connect(self.threshMean)

        self.ui.btnErode.clicked.connect(self.erosion)
        self.ui.btnDilate.clicked.connect(self.dilation)
        self.ui.btnOpen.clicked.connect(self.opening)
        self.ui.btnClose.clicked.connect(self.closing)

        self.ui.btnFindRectangle.clicked.connect(self.findRectangle)
        self.ui.btnFindMinRectangle.clicked.connect(self.findMinAreaRectangle)
        self.ui.btnFindMinCircle.clicked.connect(self.findMinCircle)
        self.ui.btnFitEllipse.clicked.connect(self.fitEllipse)

    def fileOpen(self):
        # load image file, set image to origImage, activeImage, and workingImage
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open Photo', '/home/pi/Desktop/', 'images(*.png *.jpg *.jpeg)')
        global origImage
        origImage = cv2.imread(str(file_name))
        global workingImage
        workingImage = origImage
        global activeImage
        activeImage = origImage
        global origImagePx
        origImagePx = self.mat2Pixmap(origImage, self.ui.lblOriginalImage)
        global workingImagePx
        workingImagePx = origImagePx
        self.ui.lblOriginalImage.setPixmap(origImagePx)
        activeImagePx = self.mat2Pixmap(origImage, self.ui.lblActiveImage)
        self.ui.lblActiveImage.setPixmap(activeImagePx)

    def mat2Pixmap(self, cv2Image, lbl):
        # convert image from cv mat type to qt pixmap
        qtImage = self.toQImage(cv2Image)
        # convert qt image to pixmap
        qtPixmap = QtGui.QPixmap.fromImage(qtImage)
        # scale image to fit inside window
        qtPixmap = qtPixmap.scaled(lbl.size(), QtCore.Qt.KeepAspectRatio)
        return qtPixmap


    def toQImage(self, im, copy=False):
        # converts the cv mat type image to a Qimage type
        gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]
        if im is None:
            return QtGui.QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)\
                        .rgbSwapped();
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)\
                        .rgbSwapped();
                    return qim.copy() if copy else qim

    def reset(self):
        global activeImage
        activeImage = origImage
        global workingImage
        workingImage = origImage
        self.updateActiveImg()
        self.updateWorkingImg()

    def updateActiveImg(self):
        activeImagePx = self.mat2Pixmap(activeImage,self.ui.lblActiveImage)
        self.ui.lblActiveImage.setPixmap(activeImagePx)

    def updateWorkingImg(self):
        global workingImage
        workingImage = activeImage
        global workingImagePx
        workingImagePx = self.mat2Pixmap(workingImage, self.ui.lblOriginalImage)
        self.switchMainImage()

    def switchMainImage(self):
        if not self.ui.rbOrigImg.isChecked():
            self.ui.lblOriginalImage.setPixmap(workingImagePx)
        if self.ui.rbOrigImg.isChecked():
            self.ui.lblOriginalImage.setPixmap(origImagePx)

    def toGreyscale(self, item):
        itemtext = str(item.text())
        global activeImage

        # convert image to greyscale through simple methods
        if itemtext == "Simple Greyscale":
            activeImage = cv2.cvtColor(workingImage,cv2.COLOR_BGR2GRAY)

        # extract red plane and convert to greyscale
        if itemtext in ("Blue", "Green", "Red"):
            # activeImage = workingImage[:, :, 0]
            b, g, r = cv2.split(workingImage)
            if itemtext == "Red":
                activeImage = r
            if itemtext == "Green":
                activeImage = g
            if itemtext == "Blue":
                activeImage = b

        # extract Hue, Saturation, Value
        if itemtext in ("Hue", "Saturation", "Value"):
            hsv = cv2.cvtColor(workingImage, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if itemtext == "Hue":
                activeImage = h
            if itemtext == "Saturation":
                activeImage = s
            if itemtext == "Value":
                activeImage = v
        # extract Lightness plane from HSL
        if itemtext == "Lightness":
            hsl = cv2.cvtColor(workingImage, cv2.COLOR_BGR2HSL)
            h, s, l = cv2.split(hsl)
            activeImage = l
        # exract CIELAB L* a* b*
        if itemtext in ("L*", "a*", "b*"):
            Lab = cv2.cvtColor(workingImage, cv2.COLOR_BGR2HSV)
            L, a, b = cv2.split(Lab)
            if itemtext == "L*":
                activeImage = L
            if itemtext == "a*":
                activeImage = a
            if itemtext == "b*":
                activeImage = b
        # update the active image
        self.updateActiveImg()

    def erosion(self):
        # Erode the image to get rid of noise
        kernel = np.ones((self.ui.sbKernel1.value(), self.ui.sbKernel2.value()), np.uint8)
        global activeImage
        activeImage = cv2.erode(workingImage, kernel, iterations=1)
        self.updateActiveImg()

    def dilation(self):
        kernel = np.ones((self.ui.sbKernel1.value(), self.ui.sbKernel2.value()), np.uint8)
        global activeImage
        activeImage = cv2.dilate(workingImage, kernel, iterations=1)
        self.updateActiveImg()

    def opening(self):
        kernel = np.ones((self.ui.sbKernel1.value(), self.ui.sbKernel2.value()), np.uint8)
        global activeImage
        activeImage = cv2.morphologyEx(workingImage, cv2.MORPH_OPEN, kernel)
        self.updateActiveImg()

    def closing(self):
        kernel = np.ones((self.ui.sbKernel1.value(), self.ui.sbKernel2.value()), np.uint8)
        global activeImage
        activeImage = cv2.morphologyEx(workingImage, cv2.MORPH_CLOSE, kernel)
        self.updateActiveImg()

    # Histogram equalization
    def equalizeHist(self):
        global activeImage
        activeImage = cv2.equalizeHist(workingImage)
        self.updateActiveImg()

    # Contrast Limited Adaptive Histogram Equalization
    def clahe(self):
        claheobj = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
        global activeImage
        activeImage = claheobj.apply(workingImage)
        self.updateActiveImg()

    def blurAveraging(self):
        kernel = (self.ui.sbKernel1.value(), self.ui.sbKernel2.value())
        global activeImage
        activeImage = cv2.blur(workingImage, kernel)
        self.updateActiveImg()

    def blurGaussian(self):
        kernel = (self.ui.sbKernel1.value(), self.ui.sbKernel2.value())
        global activeImage
        activeImage = cv2.GaussianBlur(workingImage, kernel, 0)
        self.updateActiveImg()

    def blurMedian(self):
        kernel = self.ui.sbKernel1.value()
        global activeImage
        activeImage = cv2.medianBlur(workingImage, kernel)
        self.updateActiveImg()

    def filterBilateral(self):
        global activeImage
        activeImage = cv2.bilateralFilter(workingImage, 9, 75, 75)
        self.updateActiveImg()

    def getThreshType(self, index):
        threshType = cv2.THRESH_BINARY
        if index == "BINARY":
            threshType = cv2.THRESH_BINARY
        if index == "BINARY_INV":
            threshType = cv2.THRESH_BINARY_INV
        if index == "TOZERO":
            threshType = cv2.THRESH_TOZERO
        if index == "TRUNC":
            threshType = cv2.THRESH_TRUNC
        if index == "TOZERO_INV":
            threshType = cv2.THRESH_TOZERO_INV
        return (threshType)

    def threshGlobal(self):
        threshType = self.getThreshType(self.ui.cbThreshType.currentText())
        global activeImage
        ret, activeImage = cv2.threshold(workingImage, self.ui.sbThreshThresh.value(),
                                         self.ui.sbThreshMaxval.value(), threshType)
        self.updateActiveImg()

    def threshMean(self):
        threshType = self.getThreshType(self.ui.cbThreshType.currentText())
        global activeImage
        activeImage = cv2.adaptiveThreshold(workingImage, self.ui.sbThreshMaxval.value(),
                                            cv2.ADAPTIVE_THRESH_MEAN_C, threshType, 11, 2)
        self.updateActiveImg()

    def threshGaussian(self):
        threshType = self.getThreshType(self.ui.cbThreshType.currentText())
        global activeImage
        activeImage = cv2.adaptiveThreshold(workingImage, self.ui.sbThreshMaxval.value(),
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshType, 11, 2)
        self.updateActiveImg()

    def auto_canny(self, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(activeImage)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        global activeImage
        activeImage = cv2.Canny(activeImage, lower, upper)
        self.updateActiveImg()

    def hsvMask(self):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(workingImage, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([self.ui.sbLH.value(), self.ui.sbLS.value(), self.ui.sbLV.value()])
        upper_blue = np.array([self.ui.sbUH.value(), self.ui.sbUS.value(), self.ui.sbUV.value()])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        global activeImage
        # activeImage = cv2.bitwise_and(workingImage,workingImage, mask= mask) #display color image
        # Uncomment above and comment the below statement to see color instead of binary as active image
        activeImage = mask  # display binarized image

        # Show the masked image
        self.updateActiveImg()

    def findContours(self):
        # Find the contours
        imgContours, contours, hierarchy = cv2.findContours(workingImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        return contours

    def findRectangle(self):
        contours = self.findContours()
        x, y, w, h = cv2.boundingRect(contours[0])
        global activeImage
        activeImage = cv2.rectangle(origImage.copy(), (x, y), (x+w, y+h), (0, 0, 255), 4)
        self.updateActiveImg()

    def findMinAreaRectangle(self):
        contours = self.findContours()
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        global activeImage
        activeImage = cv2.drawContours(origImage.copy(), [box], 0, (0, 0, 255), 4)
        self.updateActiveImg()

    def findMinCircle(self):
        contours = self.findContours()
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x), int(y))
        radius = int(radius)
        global activeImage
        activeImage = cv2.circle(origImage.copy(), center, radius, (0, 255, 0), 4)
        self.updateActiveImg()

    def fitEllipse(self):
        contours = self.findContours()
        ellipse = cv2.fitEllipse(contours[0])
        global activeImage
        activeImage = cv2.ellipse(origImage.copy(), ellipse, (0, 255, 0), 4)
        self.updateActiveImg()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
