import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import math


diseases=['SegBlast','SegSpot','SegHealthy']
label=['BLAST','SPOT']

k=1
data=[]

for j in diseases:
	print j

	segimages=os.listdir(j)

	for i in segimages:
		print i
		img = cv2.imread('./'+j+'/'+str(i))
		real_img=cv2.imread('./'+j+'/'+str(i))
		real_img=cv2.cvtColor(real_img,cv2.COLOR_BGR2GRAY)
		imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,127,255,0)

		image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		area=0
		index=0

		if(k!=3):

			for i in range(len(contours)):
				a=cv2.contourArea(contours[i])
				if(a>area):
					area=a
					index=i

			cnt=contours[index]
			M = cv2.moments(cnt)
			print M


			#ASpect ratio
			x,y,w,h = cv2.boundingRect(cnt)
			aspect_ratio = float(w)/h

			#Extent
			print w*h


			rect=cv2.minAreaRect(cnt)
			print rect[1][0]*rect[1][1]

			extent=area/(rect[1][0]*rect[1][1])
			
			#Diameter

			equi_diameter = np.sqrt(4*area/np.pi)
			print("diameter is ",equi_diameter)

			#orientaion and major and minor axis
			if(k==3):
				ma,MA=(1,0)
			else:
				(x,y),(ma,MA),angle = cv2.fitEllipse(cnt)
				print("major axis is",MA)
				print("minor axis is",ma)
				print("orientation angle is",angle)
				print "ratio",MA/ma
			#centroid

			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			print("centroid cordinates are",cX,cY)

			cord='('+str(cX)+','+str(cY)+')'

			list=[]
			list.append(format(area,'.2f'))
			list.append(format(cv2.arcLength(contours[index],True),'.2f'))
			list.append(format(aspect_ratio,'.2f'))
			list.append(format(extent,'.2f'))
			list.append(format(equi_diameter,'.2f'))
			list.append(format(MA,'.2f'))
			list.append(format(ma,'.2f'))
			list.append(format(MA/ma,'.2f'))
			list.append(len(contours))
			list.append(format(angle,'.2f'))
			list.append(format((4*area*np.pi)/pow(cv2.arcLength(contours[index],True),2),'.2f'))
			list.append(cord)
			
			list.append(k)

			
			cv2.drawContours(img, contours ,index, (0,255,0), -1)

			
		if(k==3):
			list=[]
			list=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,k]

		
		data.append(list)
		df=pd.DataFrame(data,columns=['Area','Peri','A_ratio','Extent','Dia','Major','Minor','Ratio','noc','angle','ff','cord','Label'])
	k=k+1

	#print df.to_csv(sep='\t', index=False, header=1)


	df.to_csv(r'pandas.txt', header=True, index=None, sep='\t', mode='a')

df.to_excel('Train.xlsx',sheet_name='area',index=0)
#df.to_excel('Train.xlsx',sheet_name='area',index=0,startrow=2,startcol=2)




















