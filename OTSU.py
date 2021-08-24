import numpy as np
import cv2
import matplotlib.pyplot as plt



img=cv2.imread('6.jpg')

img1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(img1)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

th=0
max_val=255

ret,out1=cv2.threshold(h,th,max_val,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,out2=cv2.threshold(h,th,max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,out3=cv2.threshold(h,th,max_val,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
ret,out4=cv2.threshold(h,th,max_val,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)


titles=['IMG','HUE','bin','bin_in','zero','zero_inv',]
images=[img,h,out1,out2,out3,out4]

for i in range(len(images)):
	plt.subplot(2,3,i+1)
	plt.title(titles[i])
	plt.imshow(images[i],cmap='gray',interpolation='bicubic')
	plt.xticks([])
	plt.yticks([])

plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpg')

img=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)


h,s,v=cv2.split(img)

# global thresholding
ret1,th1 = cv2.threshold(h,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(h,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [h, 0, th1,
          h, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
