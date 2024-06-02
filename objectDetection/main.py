from ultralytics import YOLO
import cv2
import cvzone
import math

# Colors are in BGR format, thus the color values are not in conventional format
PURPLE = (255, 0, 255)
BLUE = (255, 0, 0)

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

model = YOLO("yoloWeights/yolov8n.pt")

categories = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

run = True

people_count = 0

while run:
    success, img = camera.read()
    results = model(img, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            category = int(box.cls[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100

            cv2.rectangle(img, (x1, y1), (x2, y2), color=PURPLE, thickness=3)

            if category == categories.index("person"):
                people_count = people_count + 1

            # Parameters for the text
            text = f"{categories[category]} {confidence}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 255, 255)

            # Get the size of the text box
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = max(0, x1)
            text_y = max(35, y1)

            # Draw a rectangle for the background
            cv2.rectangle(img, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), PURPLE, cv2.FILLED)

            # Draw the text on top of the rectangle
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            print(f"Number of people is: {people_count}")

    cv2.imshow("Object Detection", img)
    cv2.waitKey(1)