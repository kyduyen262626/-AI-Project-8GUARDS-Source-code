import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
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

# Function for the main OCR Invoice Detection and Dashboard
def main():
    # Customizing the sidebar with an image logo
    st.sidebar.image('logo1.png')

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
            if uploaded_image and 'res_plotted' in locals():
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        if 'boxes' in locals():
                            for box in boxes:
                                detected_text = extract_text_from_box(box)
                                st.write(f"Detected Text: {detected_text}")
                        else:
                            st.write("No image is uploaded yet!")
                except Exception as ex:
                    st.write(f"Error: {ex}")

# Function to extract information from a detected box
def extract_info_from_box(boxes):
    extracted_info = {
        "Thông tin nhà cung cấp": {
            "Tên nhà cung cấp": "",
            "Địa chỉ nhà cung cấp": ""
        },
        "Thông tin khách hàng": {
            "Tên khách hàng": "",
            "Kho nhập hàng": "",
            "Địa chỉ khách hàng": ""
        },
        "Thông tin hóa đơn": {
            "Số hóa đơn": "",
            "Ngày nhập kho": ""
        },
        "Thông tin sản phẩm": {
            "Tên sản phẩm": "",
            "Số lượng": 0,
            "Đơn giá": 0,
            "Thành tiền": 0
        }
    }

    if box:
        # Extract information from the box and update extracted_info accordingly
        # For example:
        extracted_info["Thông tin nhà cung cấp"]["Tên nhà cung cấp"] = st.text_input("Tên nhà cung cấp")
        extracted_info["Thông tin nhà cung cấp"]["Địa chỉ nhà cung cấp"] = st.text_input("Địa chỉ nhà cung cấp")

        extracted_info["Thông tin khách hàng"]["Tên khách hàng"] = st.text_input("Tên khách hàng")
        extracted_info["Thông tin khách hàng"]["Kho nhập hàng"] = st.text_input("Kho nhập hàng")
        extracted_info["Thông tin khách hàng"]["Địa chỉ khách hàng"] = st.text_input("Địa chỉ khách hàng")

        extracted_info["Thông tin hóa đơn"]["Số hóa đơn"] = st.text_input("Số hóa đơn")
        extracted_info["Thông tin hóa đơn"]["Ngày nhập kho"] = st.date_input("Ngày nhập kho")

        extracted_info["Thông tin sản phẩm"]["Tên sản phẩm"] = st.text_input("Tên sản phẩm")
        extracted_info["Thông tin sản phẩm"]["Số lượng"] = st.number_input("Số lượng")
        extracted_info["Thông tin sản phẩm"]["Đơn giá"] = st.number_input("Đơn giá")
        extracted_info["Thông tin sản phẩm"]["Thành tiền"] = st.number_input("Thành tiền")

    return extracted_info

# Run the application 
if not st.session_state.logged_in:
    login()
else:
    main()
