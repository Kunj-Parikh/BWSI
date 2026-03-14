import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2

WIDTH, HEIGHT = 1024, 768
DISTANCE_M = 2.0
FOCAL_LENGTH_PX = 700

picam2 = Picamera2()
config = picam2.create_preview_configuration({'format':'YUV420','size':(WIDTH,HEIGHT)})
picam2.configure(config)
picam2.start()

os.makedirs("Photos", exist_ok=True)

orb = cv2.ORB_create()

def segment_light_shadow(gray):
    t,_ = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    light = gray>t
    shadow = gray<=t
    return t, light, shadow

def compare_images(img1,img2):
    k1,d1 = orb.detectAndCompute(img1,None)
    k2,d2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(d1,d2)
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    H,_ = cv2.estimateAffinePartial2D(p1,p2,method=cv2.RANSAC)
    dx,dy = H[0,2],H[1,2]
    return dx,dy, matches, k1, k2

def estimate_height(pixel_height, focal_length_px, distance_m):
    return (pixel_height * distance_m) / focal_length_px

def capture_gray():
    time.sleep(2)
    yuv = picam2.capture_array()
    gray = yuv[:HEIGHT,:WIDTH]
    filename = f"Photos/photo_{int(time.time())}.png"
    cv2.imwrite(filename, gray)
    return gray, filename

def overlay_light_shadow(gray, light, shadow, filename):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[light] = [0,255,0]
    overlay[shadow] = [0,0,255]
    cv2.imwrite(filename.replace(".png","_overlay.png"), overlay)

def overlay_height(gray, light, pixel_height, height_m, filename):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    y_coords = np.where(light)[0]
    if len(y_coords)>0:
        top = y_coords.min()
        bottom = y_coords.max()
        cv2.rectangle(overlay, (0,top), (gray.shape[1]-1,bottom), (255,0,0),2)
        cv2.putText(overlay,f"{height_m:.2f}m",(10,top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imwrite(filename.replace(".png","_height.png"), overlay)

def overlay_matches(img1,img2,matches,k1,k2,filename):
    match_img = cv2.drawMatches(img1,k1,img2,k2,matches[:20],None,flags=2)
    cv2.imwrite(filename.replace(".png","_matches.png"), match_img)

img1, file1 = capture_gray()
t1, light1, shadow1 = segment_light_shadow(img1)
y_coords = np.where(light1)[0]
pixel_height = y_coords.max() - y_coords.min() if len(y_coords)>0 else 0
height_m = estimate_height(pixel_height, FOCAL_LENGTH_PX, DISTANCE_M) if pixel_height>0 else 0
overlay_light_shadow(img1, light1, shadow1, file1)
overlay_height(img1, light1, pixel_height, height_m, file1)

time.sleep(1)

img2, file2 = capture_gray()
t2, light2, shadow2 = segment_light_shadow(img2)
dx, dy, matches, k1, k2 = compare_images(img1,img2)
overlay_light_shadow(img2, light2, shadow2, file2)
overlay_height(img2, light2, pixel_height, height_m, file2)
overlay_matches(img1,img2,matches,k1,k2,file2)

print(file1, file2)
print(t1, t2)
print(dx, dy)
print(pixel_height)
print(height_m)