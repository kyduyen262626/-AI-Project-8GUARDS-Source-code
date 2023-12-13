import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import chardet
import plotly.express as px
from ultralytics import YOLO
import pandas as pd

# Model path
model_path = 'Weight/best1.pt'

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Function for login
def login():
    st.title("8GUARDS - OCR Invoice Detection - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        # Validate login (add your own logic)
        if username == "im" and password == "123":
            st.success("Login successful!")
            # Set the session state to indicate that the user is logged in
            st.session_state.logged_in = True
            main()
        else:
            st.error("Invalid username or password")

# Function for loading the YOLO model
def load_yolo_model():
    try:
        model = YOLO(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None
    
def perform_ocr(image):
    # Existing OCR logic here
    results = []
    result = reader.readtext(image, add_margin=0.15)
    for bound in result:
        bbox = bound[0]
        x_min, y_min = [int(min(pt[0] for pt in bbox)), int(min(pt[1] for pt in bbox))]
        x_max, y_max = [int(max(pt[0] for pt in bbox)), int(max(pt[1] for pt in bbox))]
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        s = detector.predict(cropped_image)
        results.append(s)
    
    # Print or log the results
    print("OCR Results:", results)
    return results

# Function for the main OCR Invoice Detection and Dashboard
def main():

    # OCR Invoice Detection and Dashboard
    with st.sidebar:
        selected = option_menu("Home", ["Invoice Detection", 'Dashboard'],
                               icons=['body-text', 'bar-chart-line'], menu_icon="house", default_index=1)

    if selected == "Invoice Detection":
        st.title("📑OCR Invoice Detection")
        uploaded_image = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        # Set up confident
        confidence = float(st.slider(
            "Select Model Confidence", 25, 100, 40)) / 100

        model = load_yolo_model()

        # Creating two columns on the main page
        col1, col2 = st.columns(2)

        with col1:
            if uploaded_image:
                img = Image.open(uploaded_image)
                st.image(img, caption="Uploaded Image.", use_column_width=True)

                if model and st.button("Detect Object"):
                    results = model.predict(img, conf=confidence)
                    boxes = results[0].boxes
                    # Update the value of res_plotted
                    res_plotted = results[0].plot()[:, :, ::-1]

        # Creating two columns on the main page
        with col2:
            if uploaded_image and model:
                results_yolo = model.predict(img, conf=confidence)
                boxes = results_yolo[0].boxes
                res_plotted = results_yolo[0].plot()[:, :, ::-1]

                # Extract cropped images from YOLO detection
                cropped_images = []
                for i, bbox in enumerate(boxes):
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        cropped_img = img.crop((x1, y1, x2, y2))
                        cropped_images.append(cropped_img)
                    else:
                        print(f"Ignoring invalid bbox: {bbox}")

                # Initialize display_results with an empty dictionary
                display_results = {}

                # Perform OCR on each cropped image using easyocr and vietocr
                for i, cropped_img in enumerate(cropped_images):
                    if st.button(f"Detect Object {i}"):
                        easyocr_results = perform_easyocr(cropped_img)
                        vietocr_results = perform_vietocr(cropped_img)  # Replace with your VietOCR logic
                        display_results[i] = {
                            "EasyOCR": easyocr_results,
                            "VietOCR": vietocr_results
                        }
                        st.write(f"OCR Results for Cropped Image {i + 1}:", display_results[i])

                # Display the detected image and bounding boxes
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                # Display the "Detection Results" section only if OCR results are available
                with st.expander("Detection Results"):
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            text_info = display_results.get(i, {}).get("EasyOCR", "No OCR results")
                            st.write(f"Text Information for Bounding Box {i + 1}:", text_info)
                    else:
                        st.write("No bounding boxes detected.")

               
                # Run the application 
if not st.session_state.logged_in:
    login()
else:
    main()