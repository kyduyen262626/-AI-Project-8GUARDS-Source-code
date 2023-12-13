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
            if uploaded_image and 'res_plotted' in locals():
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    st.write("No image is uploaded yet!")


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

