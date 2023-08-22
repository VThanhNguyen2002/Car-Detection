# Tên môn học: Học Máy
# Mã Học Phần: 010110103604
# Lớp Học Phần: 11DHTH9
# Giảng viên hướng dẫn: Trần Đình Toàn
# Nhóm: 19

# Thư viện streamlit: 
# Streamlit là một thư viện mã nguồn mở cho phép bạn tạo các ứng dụng dữ liệu tương tác một cách nhanh chóng và dễ dàng bằng Python.
# Streamlit giúp tạo ra các ứng dụng web tương tác cho phép người dùng tương tác với dữ liệu của bạn, xem các biểu đồ, 
# trực quan hóa dữ liệu và chạy các thuật toán machine learning của bạn một cách trực quan và dễ dàng.
# Một số tính năng hữu ích của thư viện streamlit:
# Tích hợp với các thư viện dữ liệu phổ biến như pandas, matplotlib, và scikit-learn.
# Tích hợp với các công cụ machine learning như TensorFlow, PyTorch, và Keras.
# Hỗ trợ tạo giao diện người dùng trực quan bằng cách sử dụng các phương thức và tiện ích có sẵn trong thư viện.
# Cho phép trực tiếp đồng bộ hóa các thay đổi với ứng dụng của bạn trong quá trình phát triển, 
# giúp giảm thiểu thời gian phát triển và tăng tốc độ phát triển ứng dụng.
import streamlit as st
import torch # Thư viện PyTorch cho các tác vụ học sâu
import cv2 # Thư viện OpenCV cho các tác vụ xử lý ảnh
import numpy as np 
from PIL import Image # Nhập mô-đun Hình ảnh từ Thư viện Hình ảnh Python (PIL) để làm việc với hình ảnh ở nhiều định dạng khác nhau.

# Load mô hình Yolov5 bằng phương pháp load từ mô-dun torch.hub và chỉ định đường dẫn đã lưu.

@st.cache_resource
def load_model():
    '''
        Vì thư viện streamlit luôn chạy lại code mỗi khi render UI
        Nên sử dụng st.cache để tránh việc phải load lại model
    '''
    model = torch.hub.load('ultralytics/yolov5', 'custom' ,path="./best.pt", force_reload=False)
    return model
# Định nghĩa lớp: Tạo danh sách nhãn lớp cho mô hình YOLOv5.
classes = ['car']

# Tạo 1 hàm detect_objects truyền vào tham số image
# Lấy hình ảnh làm đầu vào và thực hiện phát hiện đối tượng bằng mô hình YOLOv5. 
# Hàm trả về một hình ảnh đầu ra với các hộp giới hạn và nhãn lớp được vẽ xung quanh các đối tượng được phát hiện.
# Kết quả phát hiện được trả về dưới dạng các dự đoán với tọa độ của các hộp giới hạn (bounding box), độ tin cậy (confidence) và các thông tin khác.
# Sau đó, hàm vẽ hộp giới hạn và thêm nhãn cho mỗi dự đoán và trả về ảnh với các đối tượng được đánh dấu.
def detect_objects(model, image):
    # Thực hiện phát hiện đối tượng trên hình ảnh đầu vào bằng cách sử dụng đối tượng model,là một phiên bản của mô hình YOLOv5 đã tải trước đó.
    results = model(image)
    # Trích xuất các dự đoán từ đầu ra của mô hình YOLOv5 và chuyển đổi chúng từ tenxơ PyTorch thành mảng NumPy.
    # Các dự đoán chứa thông tin về các đối tượng được phát hiện, chẳng hạn như tọa độ, ID lớp và điểm tin cậy của chúng.
    predictions = results.xyxy[0].numpy()
    # Tạo một bản sao của hình ảnh đầu vào sẽ được sửa đổi để vẽ các hộp giới hạn xung quanh các đối tượng được phát hiện.
    output_image = image.copy()
    # Lặp qua từng dự đoán trong mảng dự đoán.
    for pred in predictions:
        # Trích xuất ID lớp và điểm tin cậy từ dự đoán hiện tại.
        class_id, conf = int(pred[5]), pred[4]
        # Kiểm tra xem điểm tin cậy của dự đoán hiện tại có lớn hơn hoặc bằng 0,8 hay không. Nếu không, dự đoán sẽ bị bỏ qua.
        if conf >= 0.8:
            # Vẽ một bounding box xung quanh đối tượng được phát hiện trên tệp output_image.
            #  Các tọa độ của hộp giới hạn được trích xuất từ ​​​​dự đoán và màu sắc cũng như độ dày của hộp được chỉ định.
            cv2.rectangle(output_image, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2)
            # Thêm nhãn vào hộp giới hạn với tên của đối tượng được phát hiện.
            # ID lớp được sử dụng để tra cứu tên của đối tượng trong mảng classes 
            # và nhãn được định vị phía trên hộp giới hạn. Phông chữ, kích thước, màu sắc và độ dày của nhãn cũng được chỉ định.
            cv2.putText(output_image, classes[class_id], (int(pred[0]), int(pred[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Trả về phần đã sửa đổi output_imagevới các hộp giới hạn và nhãn được vẽ xung quanh các đối tượng được phát hiện.        
    return output_image

# Hàm app(): tạo ra ứng dụng Streamlit và định nghĩa luồng làm việc cho ứng dụng.
# Ứng dụng cho phép người dùng tải lên một ảnh và sau đó hiển thị ảnh đầu vào và ảnh đầu ra (ảnh với các đối tượng được đánh dấu).
# Kết quả đầu ra của hàm detect_objects được hiển thị trong ứng dụng bằng cách sử dụng hàm st.image() với ảnh đầu vào và ảnh đầu ra.
def app():
    # Tránh việc hiện lỗi đỏ lên UI
    with st.container():
        st.title('Bài toán nhận diện xe của Yolov5s')
        model = load_model()
        # Dòng này vô hiệu hóa cảnh báo sẽ được hiển thị nếu người dùng cố tải lên một tệp có mã hóa mà Streamlit coi là không dùng nữa.
        # st.set_option('deprecation.showfileUploaderEncoding', False)
        # Tạo ra một trình tải lên tệp trong ứng dụng Streamlit, với nhãn "Vui lòng chọn hình ảnh..." 
        # và cho phép người dùng chỉ tải lên các tệp có phần mở rộng ".jpg", ".jpeg" hoặc ".png". 
        uploaded_file = st.file_uploader("Vui lòng chọn hình ảnh...", type=["jpg", "jpeg", "png"])

        # Thao tác này kiểm tra xem một tệp đã được tải lên chưa, nghĩa là uploaded_file không phải tệp None.
        if uploaded_file is not None:
            # Đọc hình ảnh từ tệp đã tải lên và chuyển đổi nó thành một mảng có nhiều mảng bằng cách sử dụng hàm PIL.Image.open() và np.array.
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Gọi hàm detect_objects() có input image để thực hiện phát hiện đối tượng trên hình ảnh đã tải lên và lưu trữ đầu ra ở định dạng output_image.
            output_image = detect_objects(model, image)
            
            # Hiển thị cả hình ảnh đầu vào ban đầu và hình ảnh có các đối tượng được phát hiện trong ứng dụng Streamlit bằng cách sử dụng st.image()
            # Các hình ảnh được chuyển dưới dạng danh sách ([image, output_image]) với chú thích tương ứng ( ['Input Image', 'Output Image']).
            # Chiều rộng của hình ảnh được đặt thành 400 pixel bằng tham số width.
            st.image([image, output_image], caption=['Input Image', 'Output Image'], width=400)

if __name__ == '__main__':
    app()
# streamlit run UI.py