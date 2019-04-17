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

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def perspective_crop(image,w,h):
    dst1 = np.array([
        [0, 0],
        [0, h],
        [w,h],
        [w, 0]], dtype = "float32")
    points=np.array([
        [0,0],
        [0, image.shape[0]],
        [image.shape[1], image.shape[0]],
        [image.shape[1], 0]], dtype = "float32")

    M1,status = cv2.findHomography(points, dst1)
    warp1 = cv2.warpPerspective(image.copy(), M1, (w,h))
    print(np.shape(warp1))
    Back_to_size=cv2.medianBlur(warp1,3)

    return Back_to_size

def hessian(steepest_descent_matrix):
    steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    hessian=np.matmul(steepest_descent_matrix_t,steepest_descent_matrix)
    return hessian

def delta_p(hessian,steepest_descent_matrix,error):
    non_sungular=1e-15
    inv_hessian=np.linalg.inv(hessian+non_sungular*np.eye(6))
    #print(inv_hessian,'inv of hessian')
    steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    #print(np.shape(steepest_descent_matrix_t),'steepest_descent_matrix')
    error_f=np.reshape(error,(-1,1))
    A=np.matmul(steepest_descent_matrix_t,error_f)
    delta_p=np.matmul(inv_hessian,A)
    return delta_p

def gradient(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
    gradient_x=np.reshape(sobelx,(-1,1))
    sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
    gradient_y=np.reshape(sobely,(-1,1))
    gradient=[[gradient_x,gradient_y]]
    gradient=np.reshape(gradient,(-1,2))
    return gradient,sobelx,sobely

def affine_pts(m,pts):
    points=[]
    new_mat_1=np.mat([pts[0,0],pts[0,1],1]).T
    new_mat_2=np.mat([pts[1,0],pts[1,1],1]).T
    points=np.append([points],np.dot(m,new_mat_1))
    points=np.append([points],np.dot(m,new_mat_2))
    points=np.reshape(points,(2,2))
    return points.astype(int)

def convert_lab(image):
   clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
   l, a, b = cv2.split(lab)  # split on 3 different channels
   l2 = clahe.apply(l)  # apply CLAHE to the L-channel
   lab = cv2.merge((l2,a,b))  # merge channels
   img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
   return img2

def affineLKtracker(full_template,image,m,threshold,points,p):
    height,width,layers=image.shape
    diff_p=1
    old_points=points.copy()
    old_diff_p=2
    while (np.abs(diff_p-old_diff_p)>0.001):
        old_diff_p=diff_p
        print(np.abs(diff_p-old_diff_p),'difference')
        points=affine_pts(m,old_points)
        crp=image.copy()
        #cv2.imshow('image_crop1',crp)
        gray_crp=cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)
        aff_crp=nd.affine_transform(gray_crp,m)

        #step 1
        image_crop=aff_crp[points[0,1]:points[1,1],points[0,0]:points[1,0]]
        full_crop=perspective_crop(image_crop,width,height)
        #print(diff_p,'gradient')
        cv2.imshow('image_crop',full_crop)
        cv2.waitKey(10)

        #Step 3 error
        error=(full_template-full_crop)

        #Step 4 gradient
        grad_mat,img_x,img_y=gradient(image)
        aff_x=nd.affine_transform(img_x,m)
        aff_y=nd.affine_transform(img_x,m)
        gradient_roi_x=aff_x[points[0,1]:points[1,1],points[0,0]:points[1,0]]
        gradient_roi_y=aff_y[points[0,1]:points[1,1],points[0,0]:points[1,0]]
        delta_x=perspective_crop(gradient_roi_x,width,height)
        delta_y=perspective_crop(gradient_roi_y,width,height)
        gradient_x_f=np.reshape(delta_x,(-1,1))
        gradient_y_f=np.reshape(delta_y,(-1,1))
        gradient_f=[[gradient_x_f,gradient_y_f]]
        gradient_f=np.reshape(gradient_f,(-1,2))

        #Step 5 jacobian
        Jx = np.tile(np.linspace(0, width-1, width), (height, 1)).flatten()
        Jy = np.tile(np.linspace(0, height-1, height), (width, 1)).T.flatten()
        #Step 6 steepest_descent
        steepest_descent = np.vstack([gradient_f[:,0] * Jx, gradient_f[:,0] * Jy,gradient_f[:,0], gradient_f[:,1] * Jx, gradient_f[:,1] * Jy, gradient_f[:,1]]).T

        #Step 7 hessian
        hess=hessian(steepest_descent)
        #cv2.imshow('hess',hess)
        #plt.show()
        #Step 8 delta p
        p_change=delta_p(hess,steepest_descent,error)
        #print(p_change)
        diff_p=np.sum(np.multiply(p_change,p_change))
        #print(p_change*2)
        print(diff_p)
        #Step 9 Updated p
        p=p+p_change
        m=p.reshape(2,3)+I
        print(m,'changing m')
    return points
#-------------------------------------------------------------------------------
# @brief Function for pipeline for whole code
#
#  @param Path of Video
#
#  @return Array of images, size
#

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
    threshold=0.1
    image,start,end=car(150)
    #image,start,end=human(150)
    #image,start,end=vase(150)
    for i in range(start+250,end):
        image,start,end=car(i)
        #image,start,end=human(i)
        #image,start,end=vase(i)
        image=convert_lab(image)
        height,width,layers=image.shape
        size = (width,height)
        if count==0:
            global m,I
            m=np.mat([[1,0,0],[0,1,0]])
            tmp=cv2.imread('Template_car.jpg')
            #tmp=cv2.imread('Template_human.jpg')
            #tmp=cv2.imread('Template_vase.jpg')
            gray_temp=cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
            full_template=perspective_crop(gray_temp,width,height)
            points=np.mat([[122+180, 100],[341+180, 281]])
            #points=np.mat([[127,105],[333,275]])
            #points=np.mat([[265,297],[281,359]])
            #points=np.mat([[123,71],[172,151]])
            old_image=image.copy()
            p=np.zeros((6,1))
            I=m
        print(p,'p for life')
        print(m,'m for life')
        points= affineLKtracker(full_template,image,m,threshold,points,p)
        cv2.rectangle(image,(points[0,0],points[0,1]),(points[1,0],points[1,1]),(0,0,200),3)
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

#video file
# @brief Function for video processing
#
#  @param Array of images, size
#
#  @return void
#
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
