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

        # Selectbox for choosing the type of statistic
        st.sidebar.subheader("Inventory Statistics")
        selected_stat = st.sidebar.selectbox(
            "Select Type of Statistics",
            ["Inventory by Product", "Inventory Input over Time"],
        )

        # Chart based on the selected statistic
        st.subheader("Statistical Chart")

        if selected_stat == "Inventory by Product":
            # Additional options for choosing the time interval
            product_options = df['Product_name'].unique().tolist()
            product_options.insert(0, "All Products")  # Add the option "Tất cả sản phẩm" to the beginning of the list
            selected_products = st.sidebar.multiselect("Choose Products", product_options)

            # Filter DataFrame based on the selected products
            if "All Products" in selected_products:
                df_filtered = df  # Show data for all products
            else:
                df_filtered = df[df['Product_name'].isin(selected_products)]

            chart_type_options = ["Product vs Amount", "Price vs Product", "Price, Amount vs Product",
                                  "Product Pie Chart"]
            selected_chart_type = st.sidebar.radio("Choose Chart Type", chart_type_options)

            # Initialize the figure
            fig_chart = None

            # Chart based on the selected chart type
            st.subheader(
                f"Statistical chart for {', '.join(selected_products) if selected_products else 'Tất cả sản phẩm'}")

            if selected_chart_type == "Product vs Amount":
                df_amount = df_filtered.groupby('Product_name')['Amount'].sum().reset_index()
                fig_chart = px.bar(df_amount, x='Product_name', y='Amount', title='Product vs Amount',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_amount['Amount'], textposition='outside')
            elif selected_chart_type == "Price vs Product":
                df_price = df_filtered.groupby('Product_name')['Price'].sum().reset_index()
                fig_chart = px.bar(df_price, x='Product_name', y='Price', title='Price vs Product',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_price['Price'], textposition='outside')
            elif selected_chart_type == "Price, Amount vs Product":
                df_summary = df_filtered.groupby('Product_name').agg({'Amount': 'sum', 'Price': 'sum'}).reset_index()
                df_new = pd.melt(df_summary, id_vars=["Product_name"], value_vars=["Amount", "Price"],
                                 var_name='Metric', value_name='Value')
                fig_chart = px.bar(df_new, x="Product_name", y="Value", title="Price, Amount vs Product",
                                   color="Metric", barmode='group',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                # Hiển thị giá trị con số lên đỉnh của mỗi cột
                fig_chart.update_traces(text=df_new['Value'], textposition='outside')

            elif selected_chart_type == "Product Pie Chart":
                df_product_amount = df_filtered.groupby('Product_name')['Amount'].sum().reset_index()
                fig_chart = px.pie(df_product_amount, values='Amount', names='Product_name',
                                   title='Product vs Amount (Pie Chart)',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)

            # Check if the figure is not None before plotting
            if fig_chart is not None:
                st.plotly_chart(fig_chart)










        elif selected_stat == "Inventory Input over Time":
            df_filtered = pd.DataFrame()

            # Additional options for choosing the time interval
            time_interval_options = ["Date", "Month", "Quarter", "Year"]
            selected_time_interval = st.sidebar.selectbox("Choose Time Interval", time_interval_options)

            # Check if the column representing date is present in the DataFrame
            if 'Date' not in df.columns:
                st.error("The date column is not present in the DataFrame.")
            else:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

                # Filter data based on the selected time interval
                if selected_time_interval == "Date":
                    selected_date = st.sidebar.date_input("Choose Date",
                                                          min_value=df['Date'].min(),
                                                          max_value=df['Date'].max(),
                                                          value=df['Date'].min())
                    df_filtered = df[df['Date'] == selected_date]
                    title = f"Items Imported on the Day {selected_date.strftime('%d/%m/%Y')}"

                elif selected_time_interval == "Month":
                    selected_month = st.sidebar.select_slider("Choose Month", options=range(1, 13), value=1)
                    selected_year = st.sidebar.selectbox("Choose Year", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[(df['Date'].dt.month == selected_month) & (df['Date'].dt.year == selected_year)]
                    title = f"Items Imported on the Month {selected_month} in the Year {selected_year}"

                elif selected_time_interval == "Quarter":
                    selected_quarter = st.sidebar.select_slider("Choose Quarter", options=range(1, 5), value=1)
                    selected_year = st.sidebar.selectbox("Choose Year", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[
                        (df['Date'].dt.quarter == selected_quarter) & (df['Date'].dt.year == selected_year)]
                    title = f"Items Imported on the Quarter {selected_quarter} in the Year {selected_year}"

                elif selected_time_interval == "Year":
                    selected_year = st.sidebar.selectbox("Choose Year", options=df['Date'].dt.year.unique(), index=0)
                    df_filtered = df[df['Date'].dt.year == selected_year]
                    title = f"Items Imported in the Year {selected_year}"

                # Rest of the code for chart selection (Bar Chart, Pie Chart, Line Chart) goes here...

                # Additional options for choosing the products
                product_options = ["All Products"] + df['Product_name'].unique().tolist()
                selected_products = st.sidebar.multiselect("Choose Products", product_options)

                # Filter DataFrame based on the selected products
                if "All Products" not in selected_products:
                    if selected_products:
                        df_filtered = df_filtered[df_filtered['Product_name'].isin(selected_products)]
                    else:
                        st.warning("Choose at least one product.")

                # Display the filtered data
                st.write(f"{title}")
                st.write(df_filtered)

                # Chart based on the selected chart type
                # Chart based on the selected chart type
                # Chart based on the selected chart type
                st.subheader(f"Statistical chart for {title}")
                selected_chart_type = st.sidebar.selectbox("Choose Chart Type",
                                                           ["Bar Chart - Product vs Amount",
                                                            "Bar Chart - Price vs Product",
                                                            "Bar Chart - Price, Amount vs Product",
                                                            "Mixed Chart - Price, Amount vs Product"])

                if selected_chart_type == "Bar Chart - Product vs Amount":
                    fig_chart = px.bar(df_filtered, x="Product_name", y="Amount", title="Bar Chart - Product vs Amount",
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_chart.update_traces(text=df_filtered["Amount"],
                                            textposition='outside')  # Hiển thị giá trị Amount trên cột
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Bar Chart - Price vs Product":
                    fig_chart = px.bar(df_filtered, x="Product_name", y="Price", title="Bar Chart - Price vs Product",
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_chart.update_traces(text=df_filtered["Price"],textposition='outside')  # Hiển thị giá trị Price trên cột
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Bar Chart - Price, Amount vs Product":
                    fig_chart = px.bar(df_filtered, x="Product_name", y=["Amount", "Price"],
                                       title="Bar Chart - Price, Amount vs Product", barmode='group',
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_chart.update_traces(text=df_filtered["Amount"], textposition='outside',selector=dict(name='Amount'))  # Giá trị Amount
                    fig_chart.update_traces(text=df_filtered["Price"], textposition='outside',selector=dict(name='Price'))  # Giá trị Price
                    st.plotly_chart(fig_chart)

                elif selected_chart_type == "Mixed Chart - Price, Amount vs Product":
                    fig_chart = px.bar(df_filtered, x="Product_name", y="Amount",
                                       title="Mixed Chart - Price, Amount vs Product",
                                       color_discrete_sequence=px.colors.qualitative.Pastel)

                    fig_chart.add_trace(go.Scatter(x=df_filtered["Product_name"], y=df_filtered["Price"],
                                                   mode='lines', name='Price', marker=dict(color='orange')))
                    st.plotly_chart(fig_chart)


# Run the application
if not st.session_state.logged_in:
    login()
else:
    main()