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
    st.title("OCR Invoice Detection - Đăng nhập")
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
    supplier_info = {"Tên nhà cung cấp": [], "Địa chỉ nhà cung cấp": []}
    customer_info = {"Người giao": [], "Địa chỉ kho": [], "Kho": []}
    invoice_info = {"Số hóa đơn": [], "Ngày nhập kho": []}
    product_info = {"Sản phẩm": [], "Thực nhập": [], "Đơn giá": [], "Thành tiền": []}

    for result in ocr_results:
        for name, text in result.items():
            if name == "nhacungcap":
                supplier_info["Tên nhà cung cấp"].append(text)
            elif name == "diachicungcap":
                supplier_info["Địa chỉ nhà cung cấp"].append(text)
            elif name == "nguoigiao":
                customer_info["Người giao"].append(text)
            elif name == "diachikho":
                customer_info["Địa chỉ kho"].append(text)
            elif name == "kho":
                customer_info["Kho"].append(text)
            elif name == "sohoadon":
                invoice_info["Số hóa đơn"].append(text)
            elif name == "ngaynhapkho":
                invoice_info["Ngày nhập kho"].append(text)
            elif name == "sanpham":
                product_info["Sản phẩm"].append(text)
            elif name == "thucnhap":
                product_info["Thực nhập"].append(text)
            elif name == "dongia":
                product_info["Đơn giá"].append(text)
            elif name == "thanhtien":
                product_info["Thành tiền"].append(text)
    supplier_info["Tên nhà cung cấp"] = " ".join(supplier_info["Tên nhà cung cấp"])
    supplier_info["Địa chỉ nhà cung cấp"] = " ".join(supplier_info["Địa chỉ nhà cung cấp"])
    customer_info["Người giao"] = " ".join(customer_info["Người giao"])
    customer_info["Địa chỉ kho"] = " ".join(customer_info["Địa chỉ kho"])
    customer_info["Kho"] = " ".join(customer_info["Kho"])
    invoice_info["Số hóa đơn"] = " ".join(invoice_info["Số hóa đơn"])
    invoice_info["Ngày nhập kho"] = " ".join(invoice_info["Ngày nhập kho"])

    max_length = max(len(product_info[key]) for key in product_info)
    print ('max_length', max_length)
    for key in product_info:
        product_info[key] += [""] * (max_length - len(product_info[key]))

    return supplier_info, customer_info, invoice_info, product_info

# Function for the main OCR Invoice Detection and Dashboard
def main():
    # OCR Invoice Detection and Dashboard 
    with st.sidebar:
        selected = option_menu("Home", ["Invoice Detection", 'Dashboard'],
                               icons=['body-text', 'bar-chart-line'], menu_icon="house", default_index=1)

    if selected == "Invoice Detection":
        st.title("📑OCR Invoice Detection")
        uploaded_image = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png", 'bmp', 'webp'), key="image")

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

                    supplier_info, customer_info, invoice_info, product_info = extract_info_based_on_class(names, ocr_results)
                    st.session_state.supplier_info = supplier_info
                    st.session_state.customer_info = customer_info
                    st.session_state.invoice_info = invoice_info
                    st.session_state.product_info = product_info

                    print (product_info)

        # Creating two columns on the main page 
        with col2:
            if uploaded_image and 'res_plotted' in locals():
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        st.write(ocr_results)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

        # Display information in Streamlit sections
        st.header("Thông tin nhà cung cấp")
        st.session_state.supplier_info["Tên nhà cung cấp"] = st.text_input("Tên nhà cung cấp", st.session_state.supplier_info.get("Tên nhà cung cấp", ""))
        st.session_state.supplier_info["Địa chỉ nhà cung cấp"] = st.text_input("Địa chỉ nhà cung cấp", st.session_state.supplier_info.get("Địa chỉ nhà cung cấp", ""))

        st.header("Thông tin khách hàng")
        st.session_state.customer_info["Người giao"] = st.text_input("Người giao", st.session_state.customer_info.get("Người giao", ""))
        st.session_state.customer_info["Địa chỉ kho"] = st.text_input("Địa chỉ kho", st.session_state.customer_info.get("Địa chỉ kho", ""))
        st.session_state.customer_info["Kho"] = st.text_input("Kho", st.session_state.customer_info.get("Kho", ""))

        st.header("Thông tin hóa đơn")
        st.session_state.invoice_info["Số hóa đơn"] = st.text_input("Số hóa đơn", st.session_state.invoice_info.get("Số hóa đơn", ""))
        st.session_state.invoice_info["Ngày nhập kho"] = st.text_input("Ngày nhập kho", st.session_state.invoice_info.get("Ngày nhập kho", ""))

        st.header("Bảng về sản phẩm")

        # Tính toán giá trị mới cho Thực nhập và Thành tiền
        for i in range(len(st.session_state.product_info["Thực nhập"])):
            if not st.session_state.product_info["Thực nhập"][i]:
                # Nếu Thành tiền không rỗng và Đơn giá không bằng 0, tính Thực nhập
                if st.session_state.product_info["Thành tiền"][i] and float(st.session_state.product_info["Đơn giá"][i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thuc_nhap = float(st.session_state.product_info["Thành tiền"][i].replace(',', '').replace('.', '')) / float(st.session_state.product_info["Đơn giá"][i].replace(',', '').replace('.', ''))
                        st.session_state.product_info["Thực nhập"][i] = int(calculated_thuc_nhap) if calculated_thuc_nhap.is_integer() else calculated_thuc_nhap
                    except ValueError:
                        st.session_state.product_info["Thực nhập"][i] = ""

            if not st.session_state.product_info["Thành tiền"][i]:
                if st.session_state.product_info["Thực nhập"][i] and float(st.session_state.product_info["Đơn giá"][i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thanh_tien = float(st.session_state.product_info["Thực nhập"][i].replace(',', '').replace('.', '')) * float(st.session_state.product_info["Đơn giá"][i].replace(',', '').replace('.', ''))
                        st.session_state.product_info["Thành tiền"][i] = int(calculated_thanh_tien) if calculated_thanh_tien.is_integer() else calculated_thanh_tien
                    except ValueError:
                        st.session_state.product_info["Thành tiền"][i] = ""

        df_supplier = pd.DataFrame(st.session_state.supplier_info.items())
        df_customer = pd.DataFrame(st.session_state.customer_info.items())
        df_invoice = pd.DataFrame(st.session_state.invoice_info.items())
        df_products = pd.DataFrame(st.session_state.product_info)
        st.write(df_products)
        
        # Display total money 
        total_money = sum([float(amount.replace(',', '').replace('.', '')) for amount in st.session_state.product_info["Thành tiền"] if amount])
        st.write(f"Tổng thành tiền: {total_money}")

        if st.button("Save"):
            df_supplier.to_csv("extracted_information.csv", mode = "a", header=False)
            df_customer.to_csv("extracted_information.csv", mode = "a", header=False)
            df_invoice.to_csv("extracted_information.csv", mode = "a", header=False)
            df_products.to_csv("extracted_information.csv", mode = "a", header=False)
            st.success("Changes saved successfully!")

    elif selected == 'Dashboard':
        st.title("📊Dashboard")

        # Detect file encoding
        with open("2023-11-22T07-16_export.csv", 'rb') as f:
            result = chardet.detect(f.read())
        
        encoding = result['encoding']
        
        # Read CSV with detected encoding
        df = pd.read_csv("2023-11-22T07-16_export.csv", encoding=encoding)

        # Display the loaded data
        st.write("Sample Data:")
        st.write(df)

        # Selectbox for choosing the type of statistic streamlit run main.py
        st.sidebar.subheader("Thống kê kho hàng")
        selected_stat = st.sidebar.selectbox(
            "Chọn loại thống kê",
            ["Hàng tồn kho theo sản phẩm", "Hàng nhập kho theo thời gian"],
        )

        # Chart based on the selected statistic 
        st.subheader("Biểu đồ thống kê")

        if selected_stat == "Hàng tồn kho theo sản phẩm":
            # Additional options for choosing the time interval
            product_options = df['Product_name'].unique().tolist()
            product_options.insert(0, "Tất cả sản phẩm")  # Add the option "Tất cả sản phẩm" to the beginning of the list
            selected_products = st.sidebar.multiselect("Chọn sản phẩm", product_options)

            # Filter DataFrame based on the selected products
            if "Tất cả sản phẩm" in selected_products:
                df_filtered = df  # Show data for all products
            else:
                df_filtered = df[df['Product_name'].isin(selected_products)]

            chart_type_options = ["Bar Chart", "Pie Chart"]
            selected_chart_type = st.sidebar.radio("Chọn loại biểu đồ", chart_type_options)

            # Initialize the figure
            fig_chart = None

            # Chart based on the selected chart type
            st.subheader(f"Biểu đồ thống kê cho {', '.join(selected_products) if selected_products else 'Tất cả sản phẩm'}")

            if selected_chart_type == "Bar Chart":
                fig_chart = px.bar(df_filtered, x="Product_name", y="Amount", title="Bar Chart")

            elif selected_chart_type == "Pie Chart":
                fig_chart = px.pie(df_filtered, values="Amount", names="Product_name", title="Pie Chart")

            # Check if the figure is not None before plotting
            if fig_chart is not None:
                st.plotly_chart(fig_chart)

        elif selected_stat == "Hàng nhập kho theo thời gian":
            # Additional options for choosing the time interval
            time_interval_options = ["Ngày", "Tháng", "Quý", "Năm"]
            selected_time_interval = st.sidebar.selectbox("Chọn khoảng thời gian", time_interval_options)

            # Check if the column representing date is present in the DataFrame
            if 'Date' not in df.columns:
                st.error("The date column is not present in the DataFrame.")
                return
            # Parse the date column as datetime with the correct format 
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

            # Initialize the figure
            fig_chart = None

            # Additional options for choosing the chart type
            chart_type_options = ["Bar Chart", "Pie Chart", "Line Chart"]
            selected_chart_type = st.sidebar.radio("Chọn loại biểu đồ", chart_type_options)

            if selected_time_interval == "Ngày":
                # Chọn ngày từ người dùng
                selected_date = st.sidebar.date_input(
                    "Chọn ngày", 
                    min_value=df['Date'].min(), 
                    max_value=df['Date'].max(), 
                    value=df['Date'].min(),
                )
                # Lọc DataFrame theo ngày đã chọn
                df_filtered = df[df['Date'] == selected_date]
                # Tạo tiêu đề
                title = f"Hàng nhập vào ngày {selected_date.strftime('%d/%m/%Y')}"

            elif selected_time_interval == "Tháng":
                selected_month = st.sidebar.select_slider("Chọn tháng", options=range(1, 13), value=1)

                # Lấy tất cả các năm unique từ cột 'Date' trong DataFrame
                available_years = df['Date'].dt.year.unique()
                
                # Chọn năm từ user
                selected_year = st.sidebar.selectbox("Chọn năm", options=available_years, index=0)

                df_filtered = df[(df['Date'].dt.month == selected_month) & (df['Date'].dt.year == selected_year)]
                title = f"Hàng nhập vào tháng {selected_month} năm {selected_year}"

            elif selected_time_interval == "Quý":
                selected_quarter = st.sidebar.select_slider("Chọn quý", options=range(1, 5), value=1)

                # Lấy tất cả các năm unique từ cột 'Date' trong DataFrame
                available_years = df['Date'].dt.year.unique()
                    
                # Chọn năm
                selected_year = st.sidebar.selectbox("Chọn năm", options=available_years, index=0)

                df_filtered = df[(df['Date'].dt.quarter == selected_quarter) & (df['Date'].dt.year == selected_year)]
                title = f"Hàng nhập vào quý {selected_quarter} năm {selected_year}"

            elif selected_time_interval == "Năm":
                # Lấy tất cả các năm unique từ cột 'Date' trong DataFrame
                available_years = df['Date'].dt.year.unique()

                # Chọn năm
                selected_year = st.sidebar.selectbox("Chọn năm", options=available_years, index=0)

                df_filtered = df[df['Date'].dt.year == selected_year]
                title = f"Hàng nhập vào năm {selected_year}"

            # Additional options for choosing the products
            product_options = ["Tất cả sản phẩm"] + df['Product_name'].unique().tolist()
            selected_products = st.sidebar.multiselect("Chọn sản phẩm", product_options)

            # Filter DataFrame based on the selected products
            if "Tất cả sản phẩm" not in selected_products:
                if selected_products:
                    df_filtered = df_filtered[df_filtered['Product_name'].isin(selected_products)]
                else:
                    st.warning("Chọn ít nhất một sản phẩm.")

            # Display the data
            st.write(f"*{title}*")
            st.write(df_filtered)

            # Chart based on the selected chart type 
            st.subheader(f"Biểu đồ thống kê cho {title}")

            if selected_chart_type == "Bar Chart":
                fig_chart = px.bar(df_filtered, x="Product_name", y=["Amount", "Price"], title="Bar Chart",
                                color_discrete_map={"Amount": "blue", "Price": "orange"},
                                barmode='group')

            elif selected_chart_type == "Pie Chart":
                # Tạo hai biểu đồ tròn, một cho Amount và một cho Price
                fig_amount = px.pie(df_filtered, values="Amount", names="Product_name", title="Pie Chart - Amount")
                fig_price = px.pie(df_filtered, values="Price", names="Product_name", title="Pie Chart - Price")

                # Hiển thị hai biểu đồ tròn
                st.plotly_chart(fig_amount)
                st.plotly_chart(fig_price)

            elif selected_chart_type == "Line Chart":
                fig_chart = px.line(df_filtered, x="Product_name", y=["Amount", "Price"], title="Line Chart",
                                    color_discrete_map={"Amount": "blue", "Price": "orange"})

            # Check if the figure is not None before plotting
            if fig_chart is not None:
                st.plotly_chart(fig_chart)

# Run the application 
if not st.session_state.logged_in:
    login()
else:
    main()