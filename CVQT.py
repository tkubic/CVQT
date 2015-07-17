import sys
from PyQt4 import QtGui, QtCore
from CVQTui import Ui_MainWindow
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

global workingImage
workingImage = np.zeros((300, 300, 3), dtype="uint8")
global activeImage
activeImage = np.zeros((300, 300, 3), dtype="uint8")

global camera
camera = PiCamera()

class Main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupSignals()

    def setupSignals(self):

        self.ui.btnReset.clicked.connect(self.reset)
        self.ui.btnApply.clicked.connect(self.setWorkingImg)
        self.ui.btnPicamSnap.clicked.connect(self.picamSnap)

        # radio buttons under orig and active images
        self.ui.rbOrigWorking.toggled.connect(self.showOrigImage)
        self.ui.rbOrigOrig.toggled.connect(self.showOrigImage)
        self.ui.rbOrigTemplate.toggled.connect(self.showOrigImage)
        self.ui.rbActiveActive.toggled.connect(self.showActiveImage)
        self.ui.rbActiveOriginal.toggled.connect(self.showActiveImage)

        # functions under menu bar
        self.ui.actionOpen.triggered.connect(self.fileOpen)
        self.ui.actionLoad_template.triggered.connect(self.templateOpen)
        self.ui.actionSave_Image.triggered.connect(self.fileSave)

        # functions under basic tab
        self.ui.btnRotate90.clicked.connect(lambda: self.rotateImg(90))
        self.ui.btnRotate180.clicked.connect(lambda: self.rotateImg(180))
        self.ui.btnRotate270.clicked.connect(lambda: self.rotateImg(270))
        self.ui.sbRotateX.valueChanged.connect(lambda: self.rotateImg(self.ui.sbRotateX.value()))

        # functions under Region of Interest (ROI)
        self.ui.sliderROIBottom.valueChanged.connect(self.roiSelect)
        self.ui.sliderROITop.valueChanged.connect(self.roiSelect)
        self.ui.sliderROILeft.valueChanged.connect(self.roiSelect)
        self.ui.sliderROIRight.valueChanged.connect(self.roiSelect)

        # color operations
        self.ui.listGreyscale.currentItemChanged.connect(self.toGreyscale)
        self.ui.sbLH.valueChanged.connect(self.hsvMask)
        self.ui.sbLS.valueChanged.connect(self.hsvMask)
        self.ui.sbLV.valueChanged.connect(self.hsvMask)
        self.ui.sbUH.valueChanged.connect(self.hsvMask)
        self.ui.sbUS.valueChanged.connect(self.hsvMask)
        self.ui.sbUV.valueChanged.connect(self.hsvMask)

        # greyscale operations
        self.ui.btnCannyEdges.clicked.connect(self.auto_canny)
        self.ui.btnEqualizeHist.clicked.connect(self.equalizeHist)
        self.ui.btnCLAHE.clicked.connect(self.clahe)
        self.ui.btnBlurAveraging.clicked.connect(self.blurAveraging)
        self.ui.btnBlurGaussian.clicked.connect(self.blurGaussian)
        self.ui.btnBlurMedian.clicked.connect(self.blurMedian)
        self.ui.btnFilterBilateral.clicked.connect(self.filterBilateral)
        self.ui.btnThreshGlobal.clicked.connect(self.threshGlobal)
        self.ui.btnThreshGaussian.clicked.connect(self.threshGaussian)
        self.ui.btnThreshMean.clicked.connect(self.threshMean)

        # binary operations
        self.ui.btnErode.clicked.connect(self.erosion)
        self.ui.btnDilate.clicked.connect(self.dilation)
        self.ui.btnOpen.clicked.connect(self.opening)
        self.ui.btnClose.clicked.connect(self.closing)

        # find contours operations
        self.ui.btnFindRectangle.clicked.connect(self.findRectangle)
        self.ui.btnFindMinRectangle.clicked.connect(self.findMinAreaRectangle)
        self.ui.btnFindMinCircle.clicked.connect(self.findMinCircle)
        self.ui.btnFitEllipse.clicked.connect(self.fitEllipse)

        # feature detection operations
        self.ui.btnTemplateFind.clicked.connect(self.matchTemplate)

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

    def setWorkingImg(self):
        global workingImage
        print ("updating working image")
        workingImage = activeImage
        self.updateWorkingImg()

    def updateOrigImg(self):
        origImagePx = self.mat2Pixmap(origImage, self.ui.lblOrigOrig)
        self.ui.lblOrigOrig.setPixmap(origImagePx)
        self.ui.lblActiveOrig.setPixmap(origImagePx)

    def updateActiveImg(self):
        # create QT pixmap datatype for QT to display
        activeImagePx = self.mat2Pixmap(activeImage, self.ui.lblActiveActive)
        # Putting pixmap image into QT label container
        self.ui.lblActiveActive.setPixmap(activeImagePx)

    def updateWorkingImg(self):
        workingImagePx = self.mat2Pixmap(workingImage, self.ui.lblOrigWorking)
        self.ui.lblOrigWorking.setPixmap(workingImagePx)

    def updateTemplateImg(self):
        templateImagePx = self.mat2Pixmap(templateImage, self.ui.lblActiveActive)
        self.ui.lblOrigTemplate.setPixmap(templateImagePx)

    def fileOpen(self):
        # load image file, set image to origImage, activeImage, and workingImage
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open Photo', '/home/pi/Desktop/', 'images(*.png *.jpg *.jpeg)')
        global origImage
        origImage = cv2.imread(str(file_name))
        global workingImage
        workingImage = origImage
        global activeImage
        activeImage = origImage
        self.updateActiveImg()
        self.updateWorkingImg()
        self.updateOrigImg()

    def templateOpen(self):
        # load image file, set image to origImage, activeImage, and workingImage
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open Photo', '/home/pi/Desktop/', 'images(*.png *.jpg *.jpeg)')
        global templateImage
        templateImage = cv2.imread(str(file_name), 0)
        self.updateTemplateImg()

    def fileSave(self):
        # save the image currently in the workingImage variable
        file_name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '', 'images(*.png')
        file_name = str(file_name) + ".png"
        cv2.imwrite(file_name, workingImage)

    def rotateImg(self, angle):
        (h, w) = workingImage.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        global activeImage
        activeImage = cv2.warpAffine(workingImage, M, (w, h))
        self.updateActiveImg()

    def roiSelect(self):
        global activeImage
        if self.ui.rbROIEnable.isChecked():
            (h, w) = workingImage.shape[:2]
            x1 = int(self.ui.sliderROILeft.value()/100.0 * w)
            x2 = int((1 - self.ui.sliderROIRight.value()/100.0) * w)
            y1 = int(self.ui.sliderROITop.value()/100.0 * h)
            y2 = int((1 - self.ui.sliderROIBottom.value()/100.0) * h)
            print(x1, x2, y1, y2)
            if self.ui.rbROICrop.isChecked():
                activeImage = workingImage[y1:y2, x1:x2].copy()
            self.updateActiveImg()
        else:
            pass

    def reset(self):
        global activeImage
        activeImage = origImage
        global workingImage
        workingImage = origImage
        self.updateActiveImg()
        self.updateWorkingImg()

    def showOrigImage(self):
        if self.ui.rbOrigWorking.isChecked():
            self.ui.stackOrig.setCurrentIndex(0)
        elif self.ui.rbOrigOrig.isChecked():
            self.ui.stackOrig.setCurrentIndex(1)
        elif self.ui.rbOrigTemplate.isChecked():
            self.ui.stackOrig.setCurrentIndex(2)

    def showActiveImage(self):
        if self.ui.rbActiveActive.isChecked():
            self.ui.stackActive.setCurrentIndex(0)
        elif self.ui.rbActiveOriginal.isChecked():
            self.ui.stackActive.setCurrentIndex(1)

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
        # Erode the image using kernel specified
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
        v = np.median(workingImage)
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

    def matchTemplate(self):
        # sets method equal to what is in the Template Method combo box
        method = 0
        if self.ui.cbTemplateMethod.currentText() == "SQDIFF":
            method = cv2.TM_SQDIFF
        elif self.ui.cbTemplateMethod.currentText() == "SQDIFF_NORMED":
            method = cv2.TM_SQDIFF_NORMED
        elif self.ui.cbTemplateMethod.currentText() == "CCORR":
            method = cv2.TM_CCORR
        elif self.ui.cbTemplateMethod.currentText() == "CCORR_NORMED":
            method = cv2.TM_CCORR_NORMED
        elif self.ui.cbTemplateMethod.currentText() == "CCOEFF":
            method = cv2.TM_CCOEFF
        elif self.ui.cbTemplateMethod.currentText() == "CCOEFF_NORMED":
            method = cv2.TM_CCOEFF_NORMED
        # perform the cv2.matchTemplate function and store output to 'result'
        result = cv2.matchTemplate(workingImage, templateImage, method)
        # find the minimum and maximum match values as well as their locations on the photo
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else: # all other methods
            top_left = max_loc
        # height and width of template photo
        h, w = templateImage.shape[:2]
        # draw a rectangle on the active image
        bottom_right = (top_left[0] + w, top_left[1] +h)
        global activeImage
        # create a copy of the workingImage object since cv2.rectangle modifies the object directly
        # omitting the following line causes cv2.rectangle draw a rectangle on all references to the image object
        activeImage = workingImage.copy()
        cv2.rectangle(activeImage, top_left, bottom_right, 255, 4)
        self.updateActiveImg()

    def picamSnap(self):
        # initialize the camera and grab a reference to the raw camera capture
        # set camera resolution to what is selected in the combo box
        camera.resolution = map(int, str(self.ui.cbPicamResolution.currentText()).split("x"))
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        # grab an image from the camera
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        global activeImage
        activeImage = image
        global workingImage
        workingImage = image
        global origImage
        origImage = image
        self.updateActiveImg()
        self.updateWorkingImg()
        self.updateOrigImg()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
