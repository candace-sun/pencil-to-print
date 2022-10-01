# pencil-to-print

### Project Summary
> The Pencil to Print application is designed for students, acting as a simple, effective, and accessible way to convert handwritten notes to online text. The project includes a mobile application for a convenient user experience in capturing and uploading images of written work, with use of the camera on the device. This application connects to a web server that acts as the processing hub, evaluating images and classifying the text that can be identified, which are performed by computer vision and machine learning algorithms respectively. The text is then compiled and returned to the mobile device for further use.

See this [image](https://github.com/candace-sun/pencil-to-print/blob/main/Research%20Project%20Poster.png) for more information about the project.

### Background Information

As I worked on this project with a partner, I primarily worked on the web side of the app. The server was hosted on a school-specific Platform as a Service ([Director](https://director.tjhsst.edu/)), so much of the code used for server communication, involving Flask and SSH, is not easily transferrable to other services. 

**This repository serves to highlight an algorithm I developed for the project that uses computer vision to process an image of notes on notebook/lined paper and output images for each line segment identified in the image; these lines would be input into a text recognition model (see credits).**

See [this Python file](https://github.com/candace-sun/pencil-to-print/blob/main/new_preprocessing.py) for the full code. It primarily uses OpenCV (version 2).

### Function
Input: image of notes on lined paper

![Input image](https://github.com/candace-sun/pencil-to-print/blob/main/testimage0.jpg) 

Preprocessing: deskewing, bounding box detection, cropping, thresholding 

Preprocessing output: 

![Preprocessed image](https://github.com/candace-sun/pencil-to-print/blob/main/cropped.jpg)

Algorithm: uses lines of the notebook paper to identify text line segments. Applies line detection (Hough Line Transform), iterates through lines and filters out lines less likely to be from the paper. Crops output and saves each as an image.

Output: example segments produced

![Output image](https://github.com/candace-sun/pencil-to-print/blob/main/segs.png)

---
### Credits
The machine learning model used in the project was adapted from:

Scheidl, H. (2018, June 15). Build a Handwritten Text Recognition System using TensorFlow. Towards Data Science. https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
