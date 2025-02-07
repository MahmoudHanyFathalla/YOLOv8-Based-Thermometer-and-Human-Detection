# This is a python code for using the model on vs code in a local pc to detect on any local image and save it with the crops of detection  
from ultralytics import YOLO
from PIL import Image
import cv2
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO("C:\\Users\\hp\\Desktop\\Data\\py\\best.pt")

# from PIL
im1 = Image.open("C:\\Users\\hp\\Desktop\\lolo\\ooo.jpg")
results = model.predict(source=im1, show=True,save=True, save_crop=True) # saved in "C:\Users\hp\runs\detect"
cv2.waitKey(0) 
# Wait for user input before closing the window

input("Press Enter to close the window.")

