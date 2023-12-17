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
    st.session_state.supplier_info = {}
    st.session_state.customer_info = {}
    st.session_state.invoice_info = {}
    st.session_state.product_info = {}
    # OCR Invoice Detection and Dashboard 
    with st.sidebar:
        selected = option_menu("Trang chủ", ["Trích xuất", 'Thống kê'],
                               icons=['body-text', 'bar-chart-line'], menu_icon="house", default_index=1)

    if selected == "Trích xuất":
        st.title("📑Trích xuất")
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
                if model and st.button("Nhận diện hóa đơn"):
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
                    with st.expander("Kết quả nhận diện"):
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
        for i in range(len(st.session_state.product_info.get("Thực nhập", []))):
            if not st.session_state.product_info.get("Thực nhập", [])[i]:
                # Nếu Thành tiền không rỗng và Đơn giá không bằng 0, tính Thực nhập
                if st.session_state.product_info.get("Thành tiền", [])[i] and float(st.session_state.product_info.get("Đơn giá", [])[i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thuc_nhap = float(st.session_state.product_info.get("Thành tiền", [])[i].replace(',', '').replace('.', '')) / float(st.session_state.product_info.get("Đơn giá", [])[i].replace(',', '').replace('.', ''))
                        st.session_state.product_info.get("Thực nhập", [])[i] = int(calculated_thuc_nhap) if calculated_thuc_nhap.is_integer() else calculated_thuc_nhap
                    except ValueError:
                        st.session_state.product_info.get("Thực nhập", [])[i] = ""

            if not st.session_state.product_info.get("Thành tiền", [])[i]:
                if st.session_state.product_info.get("Thực nhập", [])[i] and float(st.session_state.product_info.get("Đơn giá", [])[i].replace(',', '').replace('.', '')) != 0:
                    try:
                        calculated_thanh_tien = float(st.session_state.product_info.get("Thực nhập", [])[i].replace(',', '').replace('.', '')) * float(st.session_state.product_info.get("Đơn giá", [])[i].replace(',', '').replace('.', ''))
                        st.session_state.product_info.get("Thành tiền", [])[i] = int(calculated_thanh_tien) if calculated_thanh_tien.is_integer() else calculated_thanh_tien
                    except ValueError:
                        st.session_state.product_info.get("Thành tiền", [])[i] = ""

        df_supplier = pd.DataFrame(st.session_state.supplier_info.items())
        df_customer = pd.DataFrame(st.session_state.customer_info.items())
        df_invoice = pd.DataFrame(st.session_state.invoice_info.items())
        df_products = pd.DataFrame(st.session_state.product_info)
        st.write(df_products)
        
        # Display total money 
        total_money = sum([float(amount.replace(',', '').replace('.', '')) for amount in st.session_state.product_info.get("Thành tiền", []) if amount])
        st.write(f"Tổng thành tiền: {total_money}")

        if st.button("Lưu thông tin"):
            df_supplier.to_csv("extracted_information.csv", mode = "a", header=False)
            df_customer.to_csv("extracted_information.csv", mode = "a", header=False)
            df_invoice.to_csv("extracted_information.csv", mode = "a", header=False)
            df_products.to_csv("extracted_information.csv", mode = "a", header=False)
            st.success("Lưu thành công")

    elif selected == 'Thống kê':
        st.title("📊 Thống kê")

        # Phát hiện mã hóa tệp
        with open("2023-11-22T07-16_export.csv", 'rb') as f:
            result = chardet.detect(f.read())

        encoding = result['encoding']

        # Đọc CSV với mã hóa đã phát hiện
        df = pd.read_csv("2023-11-22T07-16_export.csv", encoding=encoding)

        # Hiển thị dữ liệu đã tải
        st.write("Dữ liệu Mẫu:")
        st.write(df)

        # Dropdown để chọn loại thống kê
        st.sidebar.subheader("Thống kê kho hàng")
        selected_stat = st.sidebar.selectbox(
            "Chọn loại thống kê",
            ["Hàng nhập theo sản phẩm", "Hàng nhập kho theo thời gian"],
        )

        if selected_stat == "Hàng nhập theo sản phẩm":
            # Tùy chọn thêm để chọn khoảng thời gian
            product_options = df['Product_name'].unique().tolist()
            product_options.insert(0, "Tất cả sản phẩm")  # Thêm tùy chọn "Tất cả sản phẩm" vào đầu danh sách
            selected_products = st.sidebar.multiselect("Chọn Sản phẩm", product_options)

            # Lọc DataFrame dựa trên các sản phẩm đã chọn
            if "Tất cả sản phẩm" in selected_products:
                df_filtered = df  # Hiển thị dữ liệu cho tất cả sản phẩm
            else:
                df_filtered = df[df['Product_name'].isin(selected_products)]

            chart_type_options = ["Sản phẩm vs Số lượng", "Giá vs Sản phẩm", "Giá, Số lượng vs Sản phẩm",
                                "Biểu đồ tròn Sản phẩm"]
            selected_chart_type = st.sidebar.radio("Chọn Loại Biểu Đồ", chart_type_options)

            # Khởi tạo biểu đồ
            fig_chart = None

            if selected_chart_type == "Biểu đồ thống kê Sản phẩm và Số lượng":
                df_amount = df_filtered.groupby('Product_name')['Amount'].sum().reset_index()
                fig_chart = px.bar(df_amount, x='Product_name', y='Amount', title='Sản phẩm vs Số lượng',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_amount['Amount'], textposition='outside')
            elif selected_chart_type == "Biểu đồ thống kê Giá và Sản phẩm":
                df_price = df_filtered.groupby('Product_name')['Price'].sum().reset_index()
                fig_chart = px.bar(df_price, x='Product_name', y='Price', title='Giá vs Sản phẩm',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_price['Price'], textposition='outside')
            elif selected_chart_type == "Biểu đồ thống kê Giá, Số lượng và Sản phẩm":
                df_summary = df_filtered.groupby('Product_name').agg({'Amount': 'sum', 'Price': 'sum'}).reset_index()
                df_new = pd.melt(df_summary, id_vars=["Product_name"], value_vars=["Amount", "Price"],
                                var_name='Metric', value_name='Value')
                fig_chart = px.bar(df_new, x="Product_name", y="Value", title="Giá, Số lượng vs Sản phẩm",
                                color="Metric", barmode='group',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_new['Value'], textposition='outside')

            elif selected_chart_type == "Biểu đồ tròn Sản phẩm":
                df_product_amount = df_filtered.groupby('Product_name')['Amount'].sum().reset_index()
                fig_chart = px.pie(df_product_amount, values='Amount', names='Product_name',
                                title='Sản phẩm vs Số lượng (Biểu đồ tròn)',
                                color_discrete_sequence=px.colors.qualitative.Pastel)

            # Kiểm tra xem biểu đồ có giá trị không trước khi vẽ
            if fig_chart is not None:
                st.plotly_chart(fig_chart)

        elif selected_stat == "Hàng nhập kho theo thời gian":
            df_filtered = pd.DataFrame()

            # Tùy chọn thêm để chọn khoảng thời gian
            time_interval_options = ["Tháng", "Quý", "Năm"]  # Loại bỏ "Ngày" khỏi danh sách
            selected_time_interval = st.sidebar.selectbox("Chọn Khoảng Thời Gian", time_interval_options)

            # Kiểm tra xem cột đại diện cho ngày có trong DataFrame không
            if 'Date' not in df.columns:
                st.error("Thời gian không có trong DataFrame.")
            else:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

                # Lọc dữ liệu dựa trên khoảng thời gian đã chọn
                if selected_time_interval == "Tháng":
                    selected_month = st.sidebar.select_slider("Chọn Tháng", options=range(1, 13), value=1)
                    selected_year = st.sidebar.selectbox("Chọn Năm", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[(df['Date'].dt.month == selected_month) & (df['Date'].dt.year == selected_year)]
                    title = f"Hàng nhập trong Tháng {selected_month} năm {selected_year}"

                elif selected_time_interval == "Quý":
                    selected_quarter = st.sidebar.select_slider("Chọn Quý", options=range(1, 5), value=1)
                    selected_year = st.sidebar.selectbox("Chọn Năm", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[(df['Date'].dt.quarter == selected_quarter) & (df['Date'].dt.year == selected_year)]
                    title = f"Hàng nhập trong Quý {selected_quarter} năm {selected_year}"

                elif selected_time_interval == "Năm":
                    selected_year = st.sidebar.selectbox("Chọn Năm", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[df['Date'].dt.year == selected_year]
                    title = f"Hàng nhập trong Năm {selected_year}"
                else:
                    title = f"Hàng nhập theo thời gian"

                # Phần mã cho việc chọn loại biểu đồ (Biểu đồ Cột, Biểu đồ Tròn, Biểu đồ Đường) sẽ được thêm ở đây...

                # Tùy chọn thêm để chọn sản phẩm
                product_options = ["Tất cả sản phẩm"] + df['Product_name'].unique().tolist()
                selected_products = st.sidebar.multiselect("Chọn Sản phẩm", product_options)

                # Lọc DataFrame dựa trên các sản phẩm đã chọn
                if "Tất cả sản phẩm" not in selected_products:
                    if selected_products:
                        df_filtered = df_filtered[df_filtered['Product_name'].isin(selected_products)]
                    else:
                        st.warning("Chọn ít nhất một sản phẩm.")

                # Hiển thị dữ liệu đã lọc
                st.write(f"{title}")
                st.write(df_filtered)

                # Biểu đồ dựa trên loại biểu đồ đã chọn
                st.subheader(f"Biểu đồ thống kê cho {title}")
                selected_chart_type = st.sidebar.selectbox("Chọn Loại Biểu Đồ",
                                                        ["Biểu Đồ Cột - Sản phẩm vs Số lượng",
                                                            "Biểu Đồ Cột - Giá vs Sản phẩm",
                                                            "Biểu Đồ Cột - Giá, Số lượng vs Sản phẩm",
                                                            "Biểu Đồ Kết Hợp - Giá, Số lượng vs Sản phẩm"])

                if selected_chart_type == "Biểu Đồ Cột - Sản phẩm vs Số lượng":
                    fig_chart = px.bar(df_filtered, x="Product_name", y="Amount", title="Biểu Đồ Cột - Sản phẩm vs Số lượng",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Biểu Đồ Cột - Giá vs Sản phẩm":
                    fig_chart = px.bar(df_filtered, x="Product_name", y="Price", title="Biểu Đồ Cột - Giá vs Sản phẩm",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Biểu Đồ Cột - Giá, Số lượng vs Sản phẩm":
                    fig_chart = px.bar(df_filtered, x="Product_name", y=["Amount", "Price"],
                                    title="Biểu Đồ Cột - Giá, Số lượng vs Sản phẩm", barmode='group',
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Biểu Đồ Kết Hợp - Giá, Số lượng vs Sản phẩm":
                    fig_chart = px.bar(df_filtered, x="Product_name", y=["Amount", "Price"],
                                    title="Biểu Đồ Kết Hợp - Giá, Số lượng vs Sản phẩm", barmode='group',
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_chart)



# Chạy ứng dụng
if not st.session_state.logged_in:
    login()
else:
    main()
