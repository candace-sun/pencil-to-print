import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import cv2
from deskew import determine_skew
import time
from pathlib import Path

start = time.time()

#image = io.imread('../data/theimage.jpg')
#grayscale = rgb2gray(image)
#angle = determine_skew(grayscale)
#rotated = rotate(image, angle, resize=True) * 255
#io.imsave('output.png', rotated.astype(np.uint8))

image = cv2.imread("output.png") #test.png
#cv2.imshow("image",image)
gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)
ret,binary = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
#gray = cv2.medianBlur(gray,5)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,5)
    
#blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    

contours,hierarchy = cv2.findContours(otsu,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours:" + str(len(contours)))
contours = sorted(contours, key=cv2.contourArea, reverse= True)
x,y,w,h = cv2.boundingRect(contours[0])
print(x,y,w,h)
#cv2.drawContours(image, contours, -1, (0,255,0), 20)
#cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),30)
#cv2.imwrite("rect.jpg", image)
#exit

# Cropping an image
cropped_image = th3[y:y+h, x:x+w]
cropped_original = image[y:y+h, x:x+w]
gray = gray[y:y+h, x:x+w]
cv2.imwrite("cropped.jpg", cropped_image)

#trying to detect lines
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
cv2.imwrite("canny.jpg", edges)
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 200  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 300  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments
line_image = np.copy(cropped_image)  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

min_x = w
max_x = 0
y_vals = []

for line in lines:
    for x1,y1,x2,y2 in line:
        if (abs(y1-y2) < 5) and (abs(x1-x2) > w/40): #is a horizontal, long line
            #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            y_vals.append((y1+y2)/2)
            if(x1 < min_x): min_x = x1
            if(x2 > max_x): max_x = x2
                            
y_vals = sorted(y_vals)
ref_line_y = y_vals[0]
line_height = h

for ypos1 in y_vals:
    for ypos2 in y_vals:
        if (abs(ypos1-ypos2) > 50 and abs(ypos1-ypos2) < line_height):
            ref_line_y = min(ypos1,ypos2)
            line_height = abs(ypos1-ypos2)          

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
y = int(ref_line_y)
Path("../data/segs/").mkdir(parents=True, exist_ok=True)
f = open("segs.txt", "w")

while y > int(line_height*5):
    for seg_y in y_vals:
        if seg_y < y and seg_y > y-line_height:
            y = int(seg_y)
    cv2.line(line_image,(min_x,y),(max_x,y),(255,0,0),5)
    cv2.imwrite("../data/segs/seg"+str(int(y-line_height))+".jpg", thresh[int(y-line_height):y+15, min_x:max_x]) 
    f.write("seg"+str(int(y-line_height))+".jpg\n")
    y -= int(line_height)     
    if y < int(line_height*5):
        cv2.imwrite("../data/segs/seg"+str(int(y-line_height))+".jpg", thresh[int(line_height):int(line_height*5), min_x:max_x]) 
        f.write("seg"+str(int(y-line_height))+".jpg\n")
            
y = int(ref_line_y)
while y < h:
    for seg_y in y_vals:
        if seg_y > y and seg_y < y+line_height:
            y = int(seg_y)
    cv2.line(line_image,(min_x,y),(max_x,y),(255,0,0),5)
    cv2.imwrite("../data/segs/seg"+str(y)+".jpg", thresh[y:int(y+line_height)+15, min_x:max_x])
    f.write("seg"+str(y)+".jpg\n")
    y += int(line_height)        

f.close()
print(line_height)
print(h)
#cv2.line(line_image,(min_x,int(ref_line_y)),(max_x,int(ref_line_y)),(0,0,255),5)
#cv2.line(line_image,(min_x,int(line_height+ref_line_y)),(max_x,int(line_height+ref_line_y)),(0,0,255),5)
#lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)

end = time.time()
print(end - start)

cv2.imwrite("thresh.jpg", thresh)

#cv2.namedWindow('Lines',cv2.WINDOW_NORMAL)
#cv2.imshow('Lines', line_image)
cv2.imwrite("lines.jpg", line_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()