import cv2
import torch
import pytesseract
import sqlite3
import base64
import numpy as np
from ultralytics import YOLO  # YOLOv5 library

# Load the pre-trained YOLOv5 model
model = YOLO("yolov5s.pt")  # yolov5s is a small, fast model (you can use a custom model)

# Initialize the video capture (replace '0' with video file path if using a file)
cap = cv2.VideoCapture(0)

# Connect to SQLite database
conn = sqlite3.connect('store_products.db')
c = conn.cursor()

# Create table for storing product info
c.execute('''CREATE TABLE IF NOT EXISTS products
             (product_id INTEGER PRIMARY KEY,
              product_image BLOB,
              label_text TEXT)''')
conn.commit()

# Function to extract bounding boxes from the YOLOv5 model
def detect_products(frame):
    results = model(frame)  # Run YOLOv5 on the frame
    detections = []
    
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox[:4])  # Bounding box coordinates
            confidence = float(bbox[4])  # Confidence score
            label = int(bbox[5])  # Class label (product class)
            detections.append((x1, y1, x2, y2, confidence, label))

    return detections

# Function to extract text from a region of the image (shelf label)
def extract_text_from_label(frame, bbox):
    x1, y1, x2, y2 = bbox
    label_region = frame[y1:y2, x1:x2]  # Crop the label region from the image

    # Apply OCR using Tesseract
    text = pytesseract.image_to_string(label_region)
    return text

# Function to associate products with labels (based on spatial proximity)
def associate_product_with_label(product_bbox, label_bboxes):
    product_center = ((product_bbox[0] + product_bbox[2]) // 2, (product_bbox[1] + product_bbox[3]) // 2)
    min_distance = float('inf')
    associated_label = None

    # Loop through all labels and find the closest one
    for label_bbox in label_bboxes:
        label_center = ((label_bbox[0] + label_bbox[2]) // 2, (label_bbox[1] + label_bbox[3]) // 2)
        distance = np.linalg.norm(np.array(product_center) - np.array(label_center))

        if distance < min_distance:
            min_distance = distance
            associated_label = label_bbox

    return associated_label

# Function to store product and label information in the database
def store_in_database(frame, product_bbox, label_text):
    # Crop product image
    product_image = frame[product_bbox[1]:product_bbox[3], product_bbox[0]:product_bbox[2]]

    # Convert the image to binary for storage in the database
    _, buffer = cv2.imencode('.jpg', product_image)
    product_image_binary = base64.b64encode(buffer).decode('utf-8')

    # Insert product and label information into the database
    c.execute("INSERT INTO products (product_image, label_text) VALUES (?, ?)", 
              (product_image_binary, label_text))

    conn.commit()

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect products in the frame
    product_detections = detect_products(frame)
    
    # Detect shelf labels (for simplicity, assume they are near the bottom of the frame)
    label_detections = detect_products(frame)  # You could fine-tune another model specifically for labels

    # Loop through detected products and associate them with shelf labels
    for product_bbox in product_detections:
        associated_label_bbox = associate_product_with_label(product_bbox, label_detections)
        
        if associated_label_bbox:
            label_text = extract_text_from_label(frame, associated_label_bbox)
            store_in_database(frame, product_bbox, label_text)

            # Draw bounding boxes and label text on the detected products
            x1, y1, x2, y2, _, _ = product_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with annotations
    cv2.imshow('YOLOv5 Product Detection with OCR', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
