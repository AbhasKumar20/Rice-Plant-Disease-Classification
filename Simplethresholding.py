import numpy as np
import cv2
import matplotlib.pyplot as plt

"ACEGM0024575 "

img=cv2.imread('6.jpg')

img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

th=127
max_val=255

ret,out1=cv2.threshold(img1,th,max_val,cv2.THRESH_BINARY)
ret,out2=cv2.threshold(img1,th,max_val,cv2.THRESH_BINARY_INV)
ret,out3=cv2.threshold(img1,th,max_val,cv2.THRESH_TOZERO)
ret,out4=cv2.threshold(img1,th,max_val,cv2.THRESH_TOZERO_INV)
ret,out5=cv2.threshold(img1,th,max_val,cv2.THRESH_TRUNC)

titles=['IMG','bin','bin_in','zero','zero_inv','trunc']
images=[img,out1,out2,out3,out4,out5]

for i in range(len(images)):
	plt.subplot(2,3,i+1)
	plt.title(titles[i])
	plt.imshow(images[i],cmap='gray',interpolation='bicubic')
	plt.xticks([])
	plt.yticks([])



plt.show()