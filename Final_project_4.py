import argparse
import sys
import os, sys
import numpy as np
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
    #steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    hessian=np.dot(steepest_descent_matrix.T,steepest_descent_matrix)
    #print(hessian,'hessian')
    return hessian

def delta_p(hessian,steepest_descent_matrix,error):
    non_singular=0
    #inv_hessian=np.linalg.inv(hessian+non_singular*np.eye(6))
    inv_hessian=np.linalg.pinv(hessian)
    steepest_descent_matrix_t=np.transpose(steepest_descent_matrix)
    SD=np.dot(steepest_descent_matrix.T,error.T)
    delta_p=np.dot(inv_hessian,SD)
    return delta_p


def gradient(image):
    gray=image
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    return sobelx,sobely

def affine(m,points):
    points_result=np.dot(m,points.T)
    return points_result.T

def convert_lab(image):
   clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   l2 = clahe.apply(l)
   lab = cv2.merge((l2,a,b))
   img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   return img2

def point_matrix(points):
    a=(points[1,0]-points[0,0])+1
    b=(points[1,1]-points[0,1])+1
    #print(a)
    #print(b)
    value=a*b
    #print(value,'value')
    matrix=np.ones((3,value))
    index=0
    for Y in range(points[0,1],points[1,1]+1):
        for X in range(points[0,0],points[1,0]+1):
            matrix[0,index]=X
            matrix[1,index]=Y
            matrix[2,index]=1
            index=index+1
    #print()
    return matrix

def error_calculate(template,image,points,pts_img):
    grayImage=image
    shape=points.shape[0]
    error=np.zeros((shape,1))
    for i in range (shape):
        a=int(points[i,0])
        b=int(points[i,1])
        c=int(pts_img[i,0])
        d=int(pts_img[i,1])
        error[i,0]=template[a,b]-grayImage[c,d]
        #print(template[a,b],grayImage[c,d])
        #print(error,'error')
    return error

def affine_new(T_x_coordinates,p,points):
    x1,x2=points[0,0],points[1,0]
    y1,y2=points[0,1],points[1,1]
    vtx=np.array([[x1,x1,x2,x2],[y1,y2,y2,y1],[1,1,1,1]])
    affine_mat =np.zeros((2,3))
    count =0
    for i in range(3):
        for j in range(2):
            affine_mat[j,i]= p[count,0]
            count =count+1
    affine_mat+=w
    new_vtx=np.dot(affine_mat,vtx)
    new_pts=(np.dot(affine_mat,T_x_coordinates)).astype(int)
    return new_pts,new_vtx

def descent(sobelx,sobely,affine_coords,temp):
    sobelx_arr=img_intent.copy()
    sobely_arr=img_intent.copy()
    sobelx_arr[0,:]=sobelx[affine_coords[1,:],affine_coords[0,:]]
    sobely_arr[0,:]=sobely[affine_coords[1,:],affine_coords[0,:]]
    img1=sobelx_arr*temp[0,:]
    img2=sobely_arr*temp[0,:]
    img3=sobelx_arr*temp[1,:]
    img4=sobely_arr*temp[1,:]
    descent_img=np.vstack((img1,img2,img3,img4, sobelx_arr, sobely_arr)).T
    return descent_img

def affineLKtracker(temp,tmp_array,gray,points,p):
    diff=2
    img_x,img_y=gradient(gray)

    iter=0
    while (diff>threshold and iter<iterations):
        iter+=1
        print(p)
        print(diff)
        new_pts,new_vtx=affine_new(temp,p,points)
        #pts_img=affine(m,points)
        #print(pts_img)
        #Step 1
        new_img = img_intent.copy()
        #img_x_array = np.zeros((1,new_pts.shape[1]))
        #img_y_array = np.zeros((1,new_pts.shape[1]))

        new_img[0,:]=gray[new_pts[1,:],new_pts[0,:]]
        error=tmp_array-new_img
        descent_img=descent(img_x,img_y,new_pts,temp)

        hessian_mat=hessian(descent_img)
        deltap=delta_p(hessian_mat,descent_img,error)

        diff=np.linalg.norm(deltap)
        p = np.reshape(p,(6,1))
        p = p+deltap
    return p,new_vtx

def gray_intensity(template,image):
    #gray=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    T_mean=np.mean(template)
    I_mean=np.mean(image)
    gray=(image*(T_mean/I_mean)).astype(float)
    return gray

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

def Pipeline(start,end):
    vidObj = cv2.VideoCapture()
    count=0
    img_array=[]

    for i in range(start,end):
        if Item==1:
            image,start,end=car(i)
        if Item==2:
            image,start,end=human(i)
        if Item==3:
            image,start,end=vase(i)

        image=convert_lab(image)
        height,width,layers=image.shape
        size = (width,height)
        gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image_mean=np.mean(gray_img)
        #-----------------------------------------------------------------
        if count==0:
            #temp_mean=image_mean
            #global I
            template=image.copy()
            #w=np.mat([[1,0,0],[0,1,0]])
            #img_array=np.zeros((1,template_size))
            temp=point_matrix(points)

            p=np.zeros((6,1))
            temp_affine=(np.dot(w,temp)).astype(int)
            #tmp_array = np.zeros((1,temp_affine.shape[1]))
            tmp_array=img_intent.copy()
            tmp_array[0,:]=gray_img[temp_affine[1,:],temp_affine[0,:]]
        #------------------------------------------------------------------
        gray=gray_intensity(template,gray_img)
        #sobelx,sobely = gradient(gray)
        #count+=1
        p,new_vtx=affineLKtracker(temp,tmp_array,gray,points,p)

        Final = cv2.polylines(image,  np.int32([new_vtx.T]),  1,  (0, 0, 200),  2)
        #cv2.imshow('rect',rect_img)
        print(count)
        count += 1
        print('Frame processing index')
        print(i)
        #cv2.imwrite('%d.jpg' %count,Final)
        img_array.append(Final)
        success, image = vidObj.read()

    return img_array,size

def video(img_array,size):
    video=cv2.VideoWriter('%s.avi' %Thing,cv2.VideoWriter_fourcc(*'DIVX'), 10.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
# main
if __name__ == '__main__':

    # Calling the function
    flag=0
    while (flag==0):
        Item=int(input("Input tracking item 1:Car, 2:Human, 3:Vase\n"))
        if Item==1:
            flag=1
            points=np.mat([[122, 100],[341, 281]])
            image,start,end=car(150)
            Thing='Car'
            threshold=0.03
            iterations=1000

        elif Item==2:
            flag=1
            points=np.mat([[265,297],[281,359]])
            image,start,end=human(150)
            Thing='Human'
            threshold=0.9
            iterations=1000

        elif Item==3:
            flag=1
            points=np.mat([[100,50],[160,160]])
            image,start,end=vase(150)
            Thing='Vase'
            threshold=0.01
            iterations=100
        else:
            flag=0
            print("Wrong Input Try again, KINDLY ENTER 1 , 2 or 3 ")

    template_size=(points[1,0]-points[0,0]+1)*(points[1,1]-points[0,1]+1)
    w=np.mat([[1,0,0],[0,1,0]])
    img_intent=np.zeros((1,template_size))
    Image,size=Pipeline(start,end)
    video(Image,size)
