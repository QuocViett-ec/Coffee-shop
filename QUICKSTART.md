# ⚡ Quick Start Guide

## Cài đặt nhanh trong 3 bước

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Setup database

```bash
# Đăng nhập MySQL
mysql -u root -p

# Tạo database và import schema
mysql -u root -p < database/schema.sql
```

Hoặc:

```sql
mysql> source database/schema.sql;
```

### Bước 3: Chạy ứng dụng

```bash
python main.py
```

## 🔧 Cấu hình Database

Nếu MySQL của bạn có cấu hình khác mặc định, chỉnh sửa `utils/config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',      # Thay đổi nếu cần
    'user': 'root',           # Thay đổi username
    'password': '',           # Thêm password của bạn
    'database': 'coffee_shop',
    'port': 3306
}
```

## 👤 Tài khoản đăng ký

Đăng ký tài khoản mới qua giao diện hoặc tạo tài khoản test:

```sql
-- Password: Demo@123
INSERT INTO users (email, password_hash, full_name, membership_tier, loyalty_points)
VALUES ('demo@coffeeshop.com',
        '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92',
        'Demo User', 'Gold', 6000);
```

## 📝 Lưu ý

- Đảm bảo MySQL đang chạy trước khi start app
- Python version: 3.8+
- PyQt6 sẽ được cài tự động qua requirements.txt

## 🐛 Troubleshooting

**Lỗi kết nối database:**
- Kiểm tra MySQL service: `sudo systemctl status mysql`
- Kiểm tra thông tin đăng nhập trong `utils/config.py`
- Đảm bảo database `coffee_shop` đã được tạo

**Lỗi import PyQt6:**
```bash
pip install --upgrade PyQt6
```

**Lỗi MySQL connector:**
```bash
pip install --upgrade mysql-connector-python
```

## 📧 Support

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub repository.
