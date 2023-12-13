import streamlit as st
import cv2
from PIL import Image
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ultralytics import YOLO
import easyocr

# Load YOLO model
weights_yolo = 'Weight\\best1.pt'
yolo_model = YOLO(weights_yolo)

# Load OCR models
config_ocr = Cfg.load_config_from_name('vgg_transformer')
config_ocr['cnn']['pretrained'] = False
config_ocr['device'] = 'cpu'
ocr_detector = Predictor(config_ocr)
ocr_reader = easyocr.Reader(['vi'])

def detect_and_extract_info(image):
    # YOLO Detection
    results = yolo_model([image])
    bounding_boxes = results[0].boxes.xyxy.cpu().detach().numpy()

    # Display the image with bounding boxes
    display_image = image.copy()
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Extract text information from cropped images
    extracted_info = []
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        result_ocr = ocr_reader.readtext(cropped_img, add_margin=0.15)
        for bound in result_ocr:
            bbox_ocr = bound[0]
            x_min, y_min = [int(min(pt[0] for pt in bbox_ocr)), int(min(pt[1] for pt in bbox_ocr))]
            x_max, y_max = [int(max(pt[0] for pt in bbox_ocr)), int(max(pt[1] for pt in bbox_ocr))]
            cropped_height = y_max - y_min
            image_height = cropped_height
            cropped_image_ocr = Image.fromarray(cropped_img[y_min:y_max, x_min:x_max]).resize((image_height, image_height), Image.NEAREST)
            text_info = ocr_detector.predict(cropped_image_ocr)
            extracted_info.append(text_info)

    return display_image, extracted_info

# Streamlit App
st.title("OCR Invoice Detection")

# Column 1: Load Image
uploaded_image = st.file_uploader("Upload an invoice image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Column 2: Display Detected Boxes and Extracted Information
if st.button("Detect and Extract Information"):
    if uploaded_image is not None:
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detected_image, extracted_info = detect_and_extract_info(image_array)
        st.image(detected_image, caption="Detected Boxes", use_column_width=True)

        st.subheader("Extracted Information:")
        for info in extracted_info:
            st.write(info)
