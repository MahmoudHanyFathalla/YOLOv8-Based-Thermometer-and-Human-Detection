# This is a python code for using the model on vs code in a local pc using the laptop camera "live detecting"s
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:\\Users\\hp\\Desktop\\Data\\py\\best.pt')

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through the camera frames
while True:
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Could not read frame from camera.")
        break

# Release the camera capture object and close the display window
cap.release()
cv2.destroyAllWindows()
