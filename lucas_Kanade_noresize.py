import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
from numpy import linalg as la
from matplotlib import pyplot as plt
import math
from PIL import Image
import scipy.ndimage as nd
import random
from scipy.interpolate import RectBivariateSpline

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def hessian(steepest_descent_matrix):
    steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    hessian=np.matmul(steepest_descent_matrix_t,steepest_descent_matrix)
    return hessian

def delta_p(hessian,steepest_descent_matrix,error):
    non_singular=0
    inv_hessian=np.linalg.inv(hessian+non_singular*np.eye(6))
    steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    #error_f=np.reshape(error,(-1,1))
    A=np.matmul(steepest_descent_matrix_t,error)
    delta_p=np.matmul(inv_hessian,A)
    return delta_p

def gradient(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
    return sobelx,sobely


def affine(m,points):
    points=np.matmul(m,points.T)
    return points.T

def convert_lab(image):
   clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
   l, a, b = cv2.split(lab)  # split on 3 different channels
   l2 = clahe.apply(l)  # apply CLAHE to the L-channel
   lab = cv2.merge((l2,a,b))  # merge channels
   img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
   return img2

def point_matrix(points):
    a=int(points[1,0]-points[0,0])
    b=int(points[1,1]-points[0,1])
    value=a*b
    matrix=np.ones((value,3))
    index=0
    for i in range(points[0,1],points[0,0]):
        for j in range(points[1,1],points[1,0]):
            index=index+1
            matrix[index,0]=i
            matrix[index,1]=j
    return matrix

def error_calculate(template,image,points,pts_img):
    grayImage=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    shape=points.shape[0]
    error=np.zeros((shape,1))
    for i in range (shape):
        a=int(points[i,0])
        b=int(points[i,1])
        c=int(pts_img[i,0])
        d=int(pts_img[i,1])
        error[i,0]=template[a,b]-grayImage[c,d]
        #print(error,'error')
    return error

def affineLKtracker(template,image,points,p):
    count = 0
    m= np.zeros((2,3))
    for i in range(3):
        for j in range (2):
            m[j,i]=p[count]
            count = count +1
    m =m +I
    print(m)
    height,width,layers=image.shape
    diff_p=1
    old_points=points.copy()
    old_diff_p=2

    #Step 4 gradient and jacobian
    img_x,img_y=gradient(image)
    steepest_descent=np.zeros((len(points),6))
    i=0
    for a,b,c in points:

        one=img_x[int(a),int(b)]*int(a)
        two=img_y[int(a),int(b)]*int(a)
        three=img_x[int(a),int(b)]*int(b)
        four=img_y[int(a),int(b)]*int(b)
        five=img_x[int(a),int(b)]
        six=img_y[int(a),int(b)]
        result=np.mat([one,two,three,four,five,six])
        steepest_descent[i,:]=result
        i=i+1

    hess=hessian(steepest_descent)


    while (diff_p>1):
        print(diff_p)
        #print(p)
        #Step 1
        pts_img=affine(m,points)
        #crp=image.copy()
        #cv2.imshow('image_crop1',crp)
        #Step 3 error
        error=error_calculate(template,image,points,pts_img)
        print(np.sum(error**2),'sum')
        #print(error)
        #cv2.imshow('error',error)
        #cv2.waitKey(1000)
        p_change=delta_p(hess,steepest_descent,error)
        #diff_p=np.sqrt(np.sum(np.multiply(p_change,p_change)))
        diff_p=la.norm(p_change)
        p=p+p_change
        count = 0
        m= np.zeros((2,3))
        for i in range(3):
            for j in range (2):
                m[j,i]=p[count]
                count = count +1
        m =m +I
    return p

def point_matrix(points):
    a=int(points[1,0]-points[0,0])
    b=int(points[1,1]-points[0,1])
    value=a*b
    matrix3=np.ones((value,3))
    index=0
    for i in range(points[0,1],points[0,0]):
        for j in range(points[1,1],points[1,0]):
            index=index+1
            matrix3[index,0]=i
            matrix3[index,1]=j
    return matrix3

def car(i):
    if i<100:
        image=cv2.imread('data/car/frame00%d.jpg'%i)
    else:
        image=cv2.imread('data/car/frame0%d.jpg'%i)
    return image,20,281

def vase(i):
    if i<100:
        image=cv2.imread('data/vase/00%d.jpg'%i)
    else:
        image=cv2.imread('data/vase/0%d.jpg'%i)
    return image,19,170

def human(i):
    if i<100:
        image=cv2.imread('data/human/00%d.jpg'%i)
    else:
        image=cv2.imread('data/human/0%d.jpg'%i)
    return image,140,341

def Pipeline():
    vidObj = cv2.VideoCapture()
    count=0
    img_array=[]
    image,start,end=car(150)
    #image,start,end=human(150)
    #image,start,end=vase(150)
    for i in range(start,end):
        image,start,end=car(i)
        #image,start,end=human(i)
        #image,start,end=vase(i)
        #image=convert_lab(image)
        height,width,layers=image.shape
        size = (width,height)
        if count==0:
            global I
            temp=image
            I=np.mat([[1,0,0],[0,1,0]])
            points=np.mat([[122, 100],[341, 281]])
            points_plt=np.mat([[122, 100,1],[122, 281,1],[341, 281,1],[341, 100,1]])
            #points=np.mat([[265,297],[281,359]])
            #points=np.mat([[123,71],[172,151]])
            template=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
            old_points=points.copy()
            old_image=image.copy()
            temp=point_matrix(points)
            p=np.zeros((6,1))
            m=I
        p=affineLKtracker(template,image,temp,p)

        if count>0:
            count = 0
            m= np.zeros((2,3))
            for i in range(3):
                for j in range (2):
                    m[j,i]=p[count]
                    count = count +1
            m =m +I
        print(m)
        print(points_plt)
        points_plt_k=affine(m,points_plt)
        cv2.polylines(image,np.int32([points_plt_k]),1,(0,0,200),3)
        cv2.imshow('image',image)
        #plt.show()
        cv2.waitKey(50)

        count += 1
        print('Frame processing index')
        print(i)
        #cv2.imwrite('%d.jpg' %count,image)
        img_array.append(image)
        success, image = vidObj.read()

    return img_array,size

def video(img_array,size):
    video=cv2.VideoWriter('car.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
# main
if __name__ == '__main__':

    # Calling the function
    Image,size=Pipeline()
    video(Image,size)
