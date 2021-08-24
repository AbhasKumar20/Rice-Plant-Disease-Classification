import numpy as np
import cv2
import matplotlib.pyplot as plt




img=cv2.imread('6.jpg')
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

h,w=gray.shape

back=np.zeros((h,w),np.uint8)


ret,thresh=cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)

largest_coutour_index=0
largest_area=0

img1,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
	a=cv2.contourArea(contours[i],0)
	if(a>largest_area):
		largest_area=a
		largest_coutour_index=i



cv2.drawContours(back,contours,largest_coutour_index,(255,255,255),cv2.FILLED,8,hierarchy)

dilation_size=2
element=cv2.getStructuringElement( cv2.MORPH_RECT,
( 2*dilation_size + 1, 2*dilation_size+1 ),
  ( dilation_size, dilation_size ) )

cv2.erode(back,back,element)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv_thr=cv2.inRange(hsv,(30,10,10),(90,255,255))
dst=cv2.bitwise_not(hsv_thr, back);


plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(rgb,cmap='gray')

plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.imshow(back,cmap='gray')
plt.show()
