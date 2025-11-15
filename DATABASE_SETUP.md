# Database Setup Guide

Hướng dẫn cài đặt database cho Coffee Shop Application.

## 📋 Yêu cầu

- MySQL 5.7+ hoặc MariaDB 10.3+
- MySQL client hoặc phpMyAdmin
- Quyền tạo database

## 🚀 Cài đặt Database

### Bước 1: Tạo Database và Schema chính

```bash
mysql -u root -p < database/schema.sql
```

Hoặc trong MySQL:
```sql
SOURCE /path/to/database/schema.sql;
```

File này sẽ tạo:
- ✅ Database `coffee_shop`
- ✅ Tất cả bảng cơ bản (users, products, categories, orders, cart, vouchers, etc.)
- ✅ Sample data mẫu

### Bước 2: Chạy Updates cho tính năng mới

```bash
mysql -u root -p < database/schema_updates.sql
```

File này sẽ:
- ✅ Thêm field `icon` vào bảng `categories`
- ✅ Thêm fields `is_new`, `is_bestseller`, `is_seasonal` vào bảng `products`
- ✅ Tạo bảng `voucher_usage` (track usage voucher)
- ✅ Tạo bảng `order_status_history` (track thay đổi trạng thái đơn)
- ✅ Tạo/đảm bảo có bảng `favorites`

### Bước 3: Setup Admin Panel

```bash
./setup_admin.sh
```

Hoặc:
```bash
mysql -u root -p coffee_shop < database/admin_schema.sql
```

File này sẽ tạo:
- ✅ Bảng `admin_users` (tài khoản admin)
- ✅ Bảng `admin_activity_log` (log hoạt động admin)
- ✅ Tài khoản admin mặc định: `admin` / `admin123`

## 📊 Cấu trúc Database

### Bảng chính (Customer Side):

| Bảng | Mô tả |
|------|-------|
| `users` | Thông tin khách hàng |
| `categories` | Danh mục sản phẩm |
| `products` | Sản phẩm |
| `toppings` | Topping |
| `cart` | Giỏ hàng |
| `orders` | Đơn hàng |
| `order_items` | Chi tiết đơn hàng |
| `vouchers` | Mã giảm giá |
| `voucher_usage` | ⭐ Lịch sử sử dụng voucher |
| `favorites` | ⭐ Sản phẩm yêu thích |
| `reviews` | Đánh giá sản phẩm |
| `notifications` | Thông báo |
| `loyalty_points_history` | Lịch sử điểm tích lũy |
| `stores` | Cửa hàng |

### Bảng Admin:

| Bảng | Mô tả |
|------|-------|
| `admin_users` | Tài khoản admin |
| `admin_activity_log` | Log hoạt động admin |
| `order_status_history` | ⭐ Lịch sử thay đổi trạng thái đơn |

⭐ = Bảng mới được thêm trong `schema_updates.sql`

## 🔧 Cấu hình kết nối

Sửa file `utils/config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',  # ← Đổi password của bạn
    'database': 'coffee_shop',
    'port': 3306
}
```

## ✅ Kiểm tra

### Kiểm tra database đã tạo:

```sql
SHOW DATABASES LIKE 'coffee_shop';
```

### Kiểm tra các bảng:

```sql
USE coffee_shop;
SHOW TABLES;
```

Kết quả phải có ít nhất 20+ bảng.

### Kiểm tra fields mới:

```sql
-- Check categories có field icon
SHOW COLUMNS FROM categories LIKE 'icon';

-- Check products có fields mới
SHOW COLUMNS FROM products LIKE 'is_new';
SHOW COLUMNS FROM products LIKE 'is_bestseller';
SHOW COLUMNS FROM products LIKE 'is_seasonal';

-- Check bảng mới
SHOW TABLES LIKE 'voucher_usage';
SHOW TABLES LIKE 'order_status_history';
SHOW TABLES LIKE 'favorites';
```

### Kiểm tra admin account:

```sql
SELECT username, email, role FROM admin_users;
```

Kết quả:
```
+----------+------------------------+-------------+
| username | email                  | role        |
+----------+------------------------+-------------+
| admin    | admin@coffeeshop.com   | super_admin |
+----------+------------------------+-------------+
```

## 🐛 Troubleshooting

### Lỗi: Database already exists

```sql
DROP DATABASE coffee_shop;
-- Sau đó chạy lại schema.sql
```

### Lỗi: Table already exists

Bỏ qua lỗi này, nó an toàn vì tất cả các câu lệnh đều dùng `IF NOT EXISTS` hoặc `ADD COLUMN IF NOT EXISTS`.

### Lỗi: Foreign key constraint

Đảm bảo chạy đúng thứ tự:
1. `schema.sql` trước
2. `schema_updates.sql` sau
3. `admin_schema.sql` cuối cùng

### Lỗi kết nối từ Python

1. Kiểm tra MySQL đang chạy:
   ```bash
   sudo systemctl status mysql
   ```

2. Kiểm tra username/password trong `utils/config.py`

3. Test kết nối:
   ```bash
   python -c "from utils.database import db; print('OK' if db.test_connection() else 'FAILED')"
   ```

## 🔄 Reset Database

Nếu muốn reset toàn bộ:

```bash
# Xóa database
mysql -u root -p -e "DROP DATABASE coffee_shop;"

# Chạy lại từ đầu
mysql -u root -p < database/schema.sql
mysql -u root -p < database/schema_updates.sql
mysql -u root -p < database/admin_schema.sql
```

## 📝 Notes

- **Backup thường xuyên**: `mysqldump -u root -p coffee_shop > backup.sql`
- **Sample data** có sẵn trong `schema.sql` để test
- **Admin password mặc định**: Nhớ đổi sau lần đăng nhập đầu!
- **Production**: Tắt sample data, đổi password, tạo user MySQL riêng

## 📞 Support

Nếu gặp vấn đề, check log:
- MySQL error log: `/var/log/mysql/error.log`
- Python traceback khi chạy app
