HỆ THỐNG NHẬN DIỆN ĐỒ DÙNG HỌC TẬP BẰNG PYTHON
1. Giới thiệu đề tài
Đề tài xây dựng hệ thống nhận diện đồ dùng học tập bằng ngôn ngữ lập trình Python, sử dụng kỹ thuật xử lý ảnh và học máy để phân loại các đồ vật thông qua hình ảnh.
Hệ thống sử dụng thư viện OpenCV để tiền xử lý ảnh, tách đối tượng cần nhận diện, trích xuất đặc trưng HOG và sử dụng thuật toán SVM để phân loại đồ vật.
Các đối tượng nhận diện gồm:
- Bút bi.
- Bút chì.
- Cục tẩy.
- Thước kẻ.
2. Mục tiêu đề tài
- Tìm hiểu kiến thức cơ bản về xử lý ảnh bằng OpenCV.
- Áp dụng các phương pháp tách đối tượng khỏi nền ảnh.
- Tìm hiểu kỹ thuật trích xuất đặc trưng HOG.
- Sử dụng thuật toán SVM để phân loại đồ dùng học tập.
- Xây dựng chương trình có khả năng huấn luyện lại mô hình khi dữ liệu thay đổi.
3. Kiến thức và công nghệ sử dụng
3.1. Python
Python là ngôn ngữ lập trình được sử dụng để xây dựng chương trình nhận diện. Python có cú pháp tương đối dễ tiếp cận và hỗ trợ nhiều thư viện phục vụ xử lý ảnh, học máy.
3.2. OpenCV
OpenCV là thư viện mã nguồn mở hỗ trợ xử lý ảnh và thị giác máy tính.
Trong đề tài, OpenCV được sử dụng để:
- Đọc và xử lý hình ảnh.
- Tiền xử lý ảnh trước khi nhận diện.
- Áp dụng kỹ thuật threshold để phân tách vùng ảnh.
- Sử dụng morphology để cải thiện vùng đối tượng.
- Tìm contour nhằm xác định đường bao của vật thể.
- Chuẩn hóa hướng của đối tượng trước khi trích xuất đặc trưng.
3.3. Threshold – Phân ngưỡng ảnh
Threshold là kỹ thuật phân ngưỡng dùng để chuyển đổi hoặc phân chia các vùng ảnh dựa trên giá trị điểm ảnh.
Trong đề tài, kỹ thuật này hỗ trợ tách đồ dùng học tập khỏi nền ảnh, giúp chương trình xác định vùng có khả năng chứa đối tượng cần nhận diện.
Hiệu quả của phương pháp phụ thuộc vào độ tương phản giữa vật thể và nền, cũng như điều kiện ánh sáng.
3.4. Morphology – Phép hình thái học
Morphology là nhóm kỹ thuật xử lý ảnh thường áp dụng trên ảnh nhị phân nhằm cải thiện hình dạng của vùng đối tượng.
Kỹ thuật này có thể giúp loại bỏ các chi tiết nhiễu nhỏ, làm liền những vùng bị đứt đoạn hoặc cải thiện đường biên đối tượng.
3.5. Contour – Đường bao đối tượng
Contour là đường bao mô tả ranh giới của một vùng đối tượng trong ảnh.
Sau khi tiền xử lý và tách vật thể, chương trình sử dụng contour để xác định vùng đối tượng cần nhận diện, hỗ trợ xác định vị trí và hình dạng của vật thể.
3.6. HOG – Histogram of Oriented Gradients
HOG là phương pháp trích xuất đặc trưng dựa trên hướng và sự phân bố của gradient trong ảnh.
HOG giúp biểu diễn thông tin về hình dạng và đường biên của vật thể thành một vector đặc trưng. Vector này được sử dụng làm đầu vào cho mô hình phân loại SVM.
Trong đề tài, HOG giúp hệ thống khai thác những đặc điểm hình dạng của các đồ dùng học tập để phục vụ quá trình nhận diện.
3.7. SVM – Support Vector Machine
SVM là thuật toán học máy có giám sát được sử dụng cho bài toán phân loại dữ liệu.
Trong hệ thống, mô hình SVM được huấn luyện bằng các đặc trưng HOG trích xuất từ ảnh đồ dùng học tập đã được gán nhãn.
Sau khi huấn luyện, mô hình sử dụng đặc trưng của ảnh đầu vào để dự đoán đối tượng thuộc nhóm nào, chẳng hạn bút bi, bút chì, tẩy hoặc thước.
4. Quy trình hoạt động của hệ thống
Bước 1: Thu thập dữ liệu ảnh
Ảnh bút bi, bút chì, tẩy và thước kẻ

Bước 2: Tiền xử lý ảnh bằng OpenCV
Threshold, morphology và contour

Bước 3: Chuẩn hóa hướng vật thể
Sử dụng minAreaRect

Bước 4: Trích xuất đặc trưng
Sử dụng HOG để tạo vector đặc trưng

Bước 5: Huấn luyện và phân loại
SVM học từ dữ liệu và dự đoán nhãn đồ vật

Bước 6: Hiển thị kết quả nhận diện


5. Tăng cường dữ liệu ảnh (Data Augmentation)
Data Augmentation là kỹ thuật tạo thêm các biến thể từ ảnh gốc nhằm tăng sự đa dạng của dữ liệu huấn luyện.
Theo cấu hình mô tả trong file dự án, hệ thống sử dụng:
- Xoay ảnh với các góc \(-45^\circ\), \(-30^\circ\), \(-15^\circ\), \(+15^\circ\), \(+30^\circ\), \(+45^\circ\).
- Thay đổi độ sáng ở mức nhẹ.
Phương pháp này giúp mô hình có cơ hội học được đặc trưng của đồ vật trong nhiều hướng đặt và điều kiện ánh sáng khác nhau. Tuy nhiên, tăng cường dữ liệu không đảm bảo mô hình luôn nhận diện chính xác.
6. Tổ chức dữ liệu
Dữ liệu ảnh được chia thành các thư mục theo từng loại đồ vật.
Thư mục	Đối tượng
data/but	Bút bi
data/butchi	Bút chì
data/tay	Cục tẩy
data/thuoc	Thước kẻ
Mỗi ảnh nên chứa một vật thể chính để giảm khó khăn trong quá trình tách đối tượng và phân loại. Nên thu thập ảnh ở nhiều góc độ, vị trí và điều kiện ánh sáng khác nhau.
7. Cài đặt và chạy chương trình
Yêu cầu: Python 3.12 và các thư viện được khai báo trong requirements.txt.
Bước 1: Cài đặt thư viện
py -3.12 -m pip install -r requirements.txt


Bước 2: Chạy chương trình
py -3.12 main.py


Bước 3: Huấn luyện lại mô hình
Khi bổ sung hoặc thay đổi dữ liệu ảnh, sử dụng nút HUẤN LUYỆN LẠI trong chương trình để đọc lại bộ dữ liệu và thực hiện huấn luyện theo chức năng đã được triển khai.
8. Ưu điểm và hạn chế
Ưu điểm
- Sử dụng Python và OpenCV, phù hợp với mục tiêu học tập về xử lý ảnh.
- Kết hợp HOG và SVM để thực hiện bài toán phân loại.
- Có thể bổ sung dữ liệu để mở rộng bộ đồ dùng học tập.
- Hỗ trợ tăng cường dữ liệu bằng cách xoay ảnh và thay đổi độ sáng.
Hạn chế
- Kết quả có thể bị ảnh hưởng bởi ánh sáng, màu nền và chất lượng ảnh.
- Nếu các đồ vật có hình dạng tương tự nhau, việc phân loại có thể gặp khó khăn.
- Hiệu quả phụ thuộc vào số lượng, chất lượng và độ đa dạng của dữ liệu huấn luyện.
- Mô hình cần được kiểm thử trên ảnh thực tế để đánh giá độ chính xác.
9. Kết luận
Đề tài giúp nhóm vận dụng kiến thức Python, xử lý ảnh bằng OpenCV và học máy để xây dựng hệ thống nhận diện đồ dùng học tập. Qua quá trình thực hiện, nhóm có cơ hội tìm hiểu quy trình từ thu thập dữ liệu, tiền xử lý ảnh, trích xuất đặc trưng đến huấn luyện mô hình phân loại.
Trong tương lai, hệ thống có thể được phát triển bằng cách bổ sung nhiều loại đồ dùng học tập, cải thiện dữ liệu huấn luyện và đánh giá mô hình trên nhiều điều kiện thực tế hơn.
10. Thành viên nhóm
STT	Họ và tên	
1 Ngô Minh Đăng
2	Đinh Văn Chiến
3	Nguyễn Bá Hải
4	Nguyễn Văn Hùng
