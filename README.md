# BaiTapLon_TPTM-NNTM
**🔍 Giới thiệu đề tài:**

Trong bối cảnh các siêu thị hiện đại đang ngày càng chuyển mình theo hướng thông minh hóa và cá nhân hóa trải nghiệm mua sắm, việc ứng dụng công nghệ nhận diện khuôn mặt để gợi ý sản phẩm theo đặc điểm từng khách hàng đang trở thành xu hướng tất yếu. 
Đề tài “Hệ thống gợi ý sản phẩm trong siêu thị thông minh dựa trên nhận diện khuôn mặt” được xây dựng nhằm giúp các siêu thị vừa và nhỏ có thể ứng dụng AI để:

Tự động nhận dạng khuôn mặt khách hàng khi họ bước vào khu vực camera.

Ước lượng độ tuổi và giới tính khách hàng dựa vào hình ảnh.

Đưa ra các gợi ý sản phẩm phù hợp, kèm vị trí kệ hàng cụ thể (kệ/thứ tự/tầng).

Ghi nhận thông tin khách hàng và lượt ghé thăm để hỗ trợ phân tích hành vi.

Phân biệt khách hàng cũ và mới dựa vào ảnh khuôn mặt đã lưu.

**🧭 Sơ đồ hoạt động tổng thể:**
![image](https://github.com/user-attachments/assets/ef9d1415-61dc-4176-97a1-8f1f5b98d4d8)

🧩 Các chức năng chính của hệ thống
![image](https://github.com/user-attachments/assets/29b6612d-f14d-4ee2-a0e3-f65d84971885)

**⚙️ Công nghệ và thư viện sử dụng:**

![image](https://github.com/user-attachments/assets/60904e53-3be6-4d50-9ce5-5f673d30a240)

**🖼️ Giao diện phần mềm (GUI):**

![Screenshot 2025-06-01 093850](https://github.com/user-attachments/assets/17737da8-c85b-441f-bada-f8c359c20e06)

**✅ Ưu điểm nổi bật của hệ thống:**

Tự động hóa hoàn toàn quá trình đề xuất sản phẩm.

Không cần tương tác tay – chỉ cần khách bước vào vùng camera.

Dễ triển khai, dễ mở rộng cho nhiều camera/khu vực/kệ hàng.

Có thể dùng để thống kê hành vi người tiêu dùng, hỗ trợ marketing.



**📂 Cấu trúc thư mục:**

├── known_faces/                     # Thư mục chứa ảnh khách hàng cũ

├── main.py                          # File chính chạy hệ thống

├── history.db                       # Cơ sở dữ liệu SQLite

├── README.md                        # Tài liệu giới thiệu dự án



**🧪 2. Tính ứng dụng thực tế:**

 **Phân tích hành vi khách hàng**: Dữ liệu được lưu trong `SQLite` giúp thống kê độ tuổi, giới tính phổ biến trong ngày/tuần/tháng.

 **Hỗ trợ marketing thông minh**: Dễ dàng gợi ý sản phẩm theo giới tính, tuổi để gia tăng trải nghiệm mua sắm và tăng doanh thu.

 **Tự động hóa dịch vụ tại siêu thị**: Tăng mức độ tự động, giảm chi phí nhân sự tư vấn.

  

**🚀 Hướng phát triển tương lai:**

>  Kết nối API AI để ước lượng tuổi/giới tính chính xác hơn (VD: DeepFace, MediaPipe).

>  Phân loại nhiều người cùng lúc trong khung hình.

>  Thêm bản đồ trực quan hóa vị trí sản phẩm (dạng sơ đồ siêu thị).

>  Tích hợp camera IP, gửi cảnh báo từ xa.

>  Xây dựng hệ thống quản trị web để xem thống kê dữ liệu (Flask/Django).

**👨‍💻 Tác giả**

Nguyễn Thanh Hải

