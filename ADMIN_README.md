# Coffee Shop Admin Panel

Hệ thống quản trị cho ứng dụng Coffee Shop.

## Cài đặt

### 1. Tạo bảng Admin trong Database

Chạy script setup:
```bash
./setup_admin.sh
```

Hoặc import trực tiếp:
```bash
mysql -u root -p coffee_shop < database/admin_schema.sql
```

### 2. Tài khoản mặc định

**Username:** `admin`
**Password:** `admin123`

⚠️ **Quan trọng:** Đổi mật khẩu sau lần đăng nhập đầu tiên!

## Khởi động Admin Panel

```bash
python admin.py
```

## Tính năng

### ✅ Đã hoàn thành:

1. **Dashboard**
   - Tổng quan thống kê doanh thu
   - Số liệu đơn hàng (tổng, hôm nay, chờ xác nhận)
   - Thống kê khách hàng và sản phẩm
   - Danh sách đơn hàng gần đây

2. **Quản lý Đơn hàng**
   - Xem danh sách tất cả đơn hàng
   - Tìm kiếm đơn hàng (theo mã, tên KH, email, SĐT)
   - Lọc theo trạng thái và ngày
   - **Cập nhật trạng thái đơn hàng** (pending → confirmed → preparing → ready → delivering → completed)
   - Xem chi tiết đơn hàng
   - Hủy đơn hàng với lý do
   - Gửi thông báo tự động cho khách hàng

3. **Xác thực Admin**
   - Đăng nhập riêng cho admin
   - Phân quyền (Super Admin, Admin, Manager, Staff)
   - Ghi log hoạt động admin
   - Đổi mật khẩu

### 🚧 Đang phát triển:

4. **Quản lý Sản phẩm**
   - CRUD sản phẩm
   - Upload ảnh sản phẩm
   - Quản lý topping, giá theo size

5. **Quản lý Khách hàng**
   - Xem danh sách khách hàng
   - Xem lịch sử mua hàng
   - Quản lý tier membership

6. **Quản lý Danh mục**
   - CRUD danh mục sản phẩm
   - Sắp xếp thứ tự hiển thị

7. **Quản lý Voucher**
   - Tạo/sửa/xóa voucher
   - Thiết lập điều kiện áp dụng
   - Theo dõi usage

8. **Báo cáo**
   - Báo cáo doanh thu theo ngày/tháng/năm
   - Top sản phẩm bán chạy
   - Thống kê khách hàng
   - Export Excel/PDF

## Cấu trúc File

```
/
├── admin.py                           # Entry point cho admin panel
├── setup_admin.sh                     # Script setup database
├── database/
│   └── admin_schema.sql               # SQL schema cho admin
├── controllers/
│   ├── admin_controller.py            # Admin authentication
│   └── admin_order_controller.py      # Order management
├── views/
│   ├── admin_login_ex.py              # Admin login
│   ├── admin_main_window_ex.py        # Admin main window
│   ├── admin_dashboard_ex.py          # Dashboard
│   └── admin_orders_ex.py             # Order management
└── ui_generated/
    ├── admin_login.py                 # Login UI
    ├── admin_main_window.py           # Main window UI
    ├── admin_dashboard.py             # Dashboard UI
    └── admin_orders.py                # Orders UI
```

## Quy trình Xử lý Đơn hàng

### Trạng thái đơn hàng:

1. **⏳ Chờ xác nhận (pending)**
   - Đơn hàng mới từ khách hàng
   - Admin xem và xác nhận

2. **✅ Đã xác nhận (confirmed)**
   - Admin đã xác nhận đơn
   - Sẵn sàng pha chế

3. **👨‍🍳 Đang pha chế (preparing)**
   - Nhân viên đang pha chế
   - Khách hàng biết đơn đang được làm

4. **📦 Sẵn sàng (ready)**
   - Đơn hàng đã pha xong
   - Pickup: Khách có thể đến lấy
   - Delivery: Sẵn sàng giao

5. **🚚 Đang giao (delivering)**
   - Chỉ cho delivery
   - Shipper đang giao hàng

6. **✅ Hoàn thành (completed)**
   - Đơn hàng hoàn tất
   - Khách đã nhận hàng

7. **❌ Đã hủy (cancelled)**
   - Đơn bị hủy
   - Cần ghi rõ lý do

## Phân quyền

### 👑 Super Admin
- Toàn quyền truy cập
- Quản lý admin users
- Xem activity logs

### 🔑 Admin
- Quản lý đơn hàng
- Quản lý sản phẩm
- Quản lý khách hàng
- Xem báo cáo

### 📋 Manager
- Quản lý đơn hàng
- Quản lý sản phẩm
- Xem báo cáo cơ bản

### 👤 Staff
- Xem và cập nhật trạng thái đơn hàng
- Xem danh sách sản phẩm

## Thông báo cho Khách hàng

Khi admin cập nhật trạng thái đơn hàng, hệ thống tự động tạo thông báo cho khách hàng:

- **Đã xác nhận:** "Đơn hàng #123 đã được xác nhận"
- **Đang pha chế:** "Đơn hàng #123 đang được pha chế"
- **Sẵn sàng:** "Đơn hàng #123 đã sẵn sàng để lấy"
- **Đang giao:** "Đơn hàng #123 đang được giao"
- **Hoàn thành:** "Đơn hàng #123 đã hoàn thành"
- **Đã hủy:** "Đơn hàng #123 đã bị hủy"

## Troubleshooting

### Không thể đăng nhập
- Kiểm tra database đã có bảng `admin_users` chưa
- Chạy lại `setup_admin.sh`
- Kiểm tra username/password

### Không thấy đơn hàng
- Kiểm tra filter trạng thái và ngày
- Đảm bảo có đơn hàng trong database
- Thử refresh (nút 🔄)

### Không cập nhật được trạng thái
- Kiểm tra admin đã đăng nhập chưa
- Kiểm tra quyền của admin account
- Xem log để debug

## Support

Nếu có vấn đề, check log trong terminal hoặc liên hệ dev team.
