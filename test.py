import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import chardet
import plotly.express as px
from ultralytics import YOLO
import pandas as pd
import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'

detector = Predictor(config)

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

reader = easyocr.Reader(['vi'])

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
    
def extract_info_based_on_class(names, ocr_results):
    supplier_info = {"Tên nhà cung cấp": "", "Địa chỉ nhà cung cấp": ""}
    customer_info = {"Người giao": "", "Địa chỉ kho": "", "Kho": ""}
    invoice_info = {"Số hóa đơn": "", "Ngày nhập kho": ""}
    product_info = {"Sản phẩm": [], "Thực nhập": [], "Đơn giá": [], "Thành tiền": []}

    max_length = max(len(product_info[key]) for key in product_info)

    for result in ocr_results:
        for name, text in result.items():
            if name == "nhacungcap":
                supplier_info["Tên nhà cung cấp"] = text
            elif name == "diachicungcap":
                supplier_info["Địa chỉ nhà cung cấp"] = text
            elif name == "nguoigiao":
                customer_info["Người giao"] = text
            elif name == "diachikho":
                customer_info["Địa chỉ kho"] = text
            elif name == "kho":
                customer_info["Kho"] = text
            elif name == "sohoadon":
                invoice_info["Số hóa đơn"] = text
            elif name == "ngaynhapkho":
                invoice_info["Ngày nhập kho"] = text
            elif name == "sanpham":
                product_info["Sản phẩm"].append(text)
            elif name == "thucnhap":
                product_info["Thực nhập"].append(text)
            elif name == "dongia":
                product_info["Đơn giá"].append(text)
            elif name == "thanhtien":
                product_info["Thành tiền"].append(text)

    # Pad lists with empty strings to make them of equal length
    for key in product_info:
        product_info[key] += [""] * (max_length - len(product_info[key]))

    return supplier_info, customer_info, invoice_info, product_info

# Function for the main OCR Invoice Detection and Dashboard
def main():
    # OCR Invoice Detection and Dashboard 
    with st.sidebar:
        selected = option_menu("Home", ["Invoice Detection", 'Dashboard'],
                               icons=['body-text', 'bar-chart-line'], menu_icon="house", default_index=1)

    names = []  
    ocr_results = []

    if selected == "Invoice Detection":
        st.title("📑OCR Invoice Detection")
        uploaded_image = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        # Set up confidence
        confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

        model = load_yolo_model()

        # Creating two columns on the main page
        col1, col2 = st.columns(2)

        with col1:
            if uploaded_image:
                img = Image.open(uploaded_image)
                st.image(img, caption="Uploaded Image.", use_column_width=True)

                # Lấy box -> model
                if model and st.button("Detect Object"):
                    results = model.predict(img, conf=confidence)
                    names = results[0].names
                    boxes = results[0].boxes
                    classes = results[0].boxes.cls
                    print("Detected Classes:", classes)
                    print("Class Names:", names)
                    # Update the value of res_plotted 
                    res_plotted = results[0].plot()[:, :, ::-1]

                    cropped_images = []
                    for bbox, cls in zip(boxes, classes):
                        bounding_box = tensor_to_numpy(bbox.xyxy)
                        x1, y1, x2, y2 = bounding_box[0]
                        # Crop the image to the bounding box, note that numpy uses y first
                        cropped_img = results[0].orig_img[int(y1):int(y2), int(x1):int(x2)]
                        cropped_images.append((cropped_img, cls))

                    ocr_results = []
                    for i in range(len(cropped_images)):
                        result = reader.readtext(cropped_images[i][0], add_margin=0.15)
                        for bound in result:
                            bbox = bound[0]
                            # Crop the image to the bounding box
                            x_min, y_min = [int(min(pt[0] for pt in bbox)), int(min(pt[1] for pt in bbox))]
                            x_max, y_max = [int(max(pt[0] for pt in bbox)), int(max(pt[1] for pt in bbox))]
                            cropped_image = Image.fromarray(cropped_images[i][0][y_min:y_max, x_min:x_max])
                            s = detector.predict(cropped_image)
                            ocr_results.append({names[int(cropped_images[i][1])]: s})

        # Creating two columns on the main page
        with col2:
            if uploaded_image and 'res_plotted' in locals():
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        st.write(ocr_results)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

        # Extract information based on class names
        supplier_info, customer_info, invoice_info, product_info = extract_info_based_on_class(names, ocr_results)

        # Display information in Streamlit sections
        st.header("Thông tin nhà cung cấp")
        supplier_info["Tên nhà cung cấp"] = st.text_input("Tên nhà cung cấp", supplier_info["Tên nhà cung cấp"])
        supplier_info["Địa chỉ nhà cung cấp"] = st.text_input("Địa chỉ nhà cung cấp", supplier_info["Địa chỉ nhà cung cấp"])

        st.header("Thông tin khách hàng")
        customer_info["Người giao"] = st.text_input("Người giao", customer_info["Người giao"])
        customer_info["Địa chỉ kho"] = st.text_input("Địa chỉ kho", customer_info["Địa chỉ kho"])
        customer_info["Kho"] = st.text_input("Kho", customer_info["Kho"])

        st.header("Thông tin hóa đơn")
        invoice_info["Số hóa đơn"] = st.text_input("Số hóa đơn", invoice_info["Số hóa đơn"])
        invoice_info["Ngày nhập kho"] = st.text_input("Ngày nhập kho", invoice_info["Ngày nhập kho"])

        st.header("Bảng về sản phẩm")

        # Remove unnecessary split() calls
        product_info["Sản phẩm"] = [item.strip() for item in product_info["Sản phẩm"] if item.strip()]
        product_info["Thực nhập"] = [item.strip() for item in product_info["Thực nhập"] if item.strip()]
        product_info["Đơn giá"] = [item.strip() for item in product_info["Đơn giá"] if item.strip()]
        product_info["Thành tiền"] = [item.strip() for item in product_info["Thành tiền"] if item.strip()]

        # Pad lists with empty strings to make them of equal length
        max_length = max(len(product_info[key]) for key in product_info)
        for key in product_info:
            product_info[key] += [""] * (max_length - len(product_info[key]))

        # Tính toán giá trị mới cho Thực nhập và Thành tiền
        for i in range(len(product_info["Thực nhập"])):
            if not product_info["Thực nhập"][i]:
                # Nếu Thành tiền không rỗng và Đơn giá không bằng 0, tính Thực nhập
                if product_info["Thành tiền"][i] and float(product_info["Đơn giá"][i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thuc_nhap = float(product_info["Thành tiền"][i].replace(',', '').replace('.', '')) / float(product_info["Đơn giá"][i].replace(',', '').replace('.', ''))
                        product_info["Thực nhập"][i] = int(calculated_thuc_nhap) if calculated_thuc_nhap.is_integer() else calculated_thuc_nhap
                    except ValueError:
                        product_info["Thực nhập"][i] = ""

            if not product_info["Thành tiền"][i]:
                # Nếu Thực nhập không rỗng và Đơn giá không bằng 0, tính Thành tiền
                if product_info["Thực nhập"][i] and float(product_info["Đơn giá"][i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thanh_tien = float(product_info["Thực nhập"][i].replace(',', '').replace('.', '')) * float(product_info["Đơn giá"][i].replace(',', '').replace('.', ''))
                        product_info["Thành tiền"][i] = int(calculated_thanh_tien) if calculated_thanh_tien.is_integer() else calculated_thanh_tien
                    except ValueError:
                        product_info["Thành tiền"][i] = ""

        # Display the final DataFrame
        df_products = pd.DataFrame(product_info)
        st.write(df_products)

        # Display total money
        total_money = sum([float(amount.replace(',', '').replace('.', '')) for amount in product_info["Thành tiền"] if amount])
        st.write(f"Tổng thành tiền: {total_money}")

        if st.button("Save Changes"):
            st.success("Changes saved successfully!")

# Run the application 
if not st.session_state.logged_in:
    login()
else:
    main()