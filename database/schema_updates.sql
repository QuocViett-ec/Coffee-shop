-- Coffee Shop Database Schema Updates
-- Updates for new features (Favorites, Admin Management, etc.)
-- Compatible with MySQL 5.7+ and MariaDB 10.0+

USE coffee_shop;

-- ============================================
-- 1. UPDATE CATEGORIES TABLE
-- ============================================

-- Add 'icon' field for emoji icons
SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='icon') > 0,
    "SELECT 1",
    "ALTER TABLE categories ADD COLUMN icon VARCHAR(10) DEFAULT '☕' AFTER description"
));
PREPARE stmt FROM @s;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add updated_at field
SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='updated_at') > 0,
    "SELECT 1",
    "ALTER TABLE categories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
));
PREPARE stmt FROM @s;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ============================================
-- 2. UPDATE PRODUCTS TABLE
-- ============================================

-- Add is_new field
SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='products' AND COLUMN_NAME='is_new') > 0,
    "SELECT 1",
    "ALTER TABLE products ADD COLUMN is_new BOOLEAN DEFAULT FALSE AFTER is_featured"
));
PREPARE stmt FROM @s;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add is_bestseller field
SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='products' AND COLUMN_NAME='is_bestseller') > 0,
    "SELECT 1",
    "ALTER TABLE products ADD COLUMN is_bestseller BOOLEAN DEFAULT FALSE AFTER is_new"
));
PREPARE stmt FROM @s;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add is_seasonal field
SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='products' AND COLUMN_NAME='is_seasonal') > 0,
    "SELECT 1",
    "ALTER TABLE products ADD COLUMN is_seasonal BOOLEAN DEFAULT FALSE AFTER is_bestseller"
));
PREPARE stmt FROM @s;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ============================================
-- 3. CREATE VOUCHER_USAGE TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS voucher_usage (
    id INT AUTO_INCREMENT PRIMARY KEY,
    voucher_id INT NOT NULL,
    user_id INT NOT NULL,
    order_id INT,
    discount_amount DECIMAL(10, 2) NOT NULL,
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (voucher_id) REFERENCES vouchers(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL,
    INDEX idx_voucher (voucher_id),
    INDEX idx_user (user_id),
    INDEX idx_used_at (used_at)
) ENGINE=InnoDB;

-- ============================================
-- 4. CREATE ORDER_STATUS_HISTORY TABLE
-- ============================================

-- Note: This requires admin_users table to exist
-- Run admin_schema.sql first, or this will fail on FK constraint

SET @table_exists = (
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = DATABASE() AND table_name = 'order_status_history'
);

SET @admin_table_exists = (
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = DATABASE() AND table_name = 'admin_users'
);

-- Create table without FK if admin_users doesn't exist
SET @sql_create = IF(@table_exists = 0,
    IF(@admin_table_exists = 0,
        -- Without FK to admin_users
        "CREATE TABLE order_status_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT NOT NULL,
            old_status ENUM('pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled'),
            new_status ENUM('pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled') NOT NULL,
            changed_by_admin_id INT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
            INDEX idx_order (order_id),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB",
        -- With FK to admin_users
        "CREATE TABLE order_status_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT NOT NULL,
            old_status ENUM('pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled'),
            new_status ENUM('pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled') NOT NULL,
            changed_by_admin_id INT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
            FOREIGN KEY (changed_by_admin_id) REFERENCES admin_users(id) ON DELETE SET NULL,
            INDEX idx_order (order_id),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB"
    ),
    "SELECT 1"
);

PREPARE stmt FROM @sql_create;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ============================================
-- 5. CREATE FAVORITES TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS favorites (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_product (user_id, product_id),
    INDEX idx_user (user_id),
    INDEX idx_product (product_id)
) ENGINE=InnoDB;

-- ============================================
-- 6. UPDATE SAMPLE DATA
-- ============================================

-- Update existing products with new fields (only if they exist)
UPDATE products SET is_new = TRUE WHERE id IN (1, 2, 3) LIMIT 3;
UPDATE products SET is_bestseller = TRUE WHERE id IN (1, 4, 5) LIMIT 3;
UPDATE products SET is_seasonal = FALSE;

-- Update categories with icons (only if column exists and categories exist)
UPDATE categories SET icon = '☕' WHERE id = 1 AND EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='icon');
UPDATE categories SET icon = '🥤' WHERE id = 2 AND EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='icon');
UPDATE categories SET icon = '🍰' WHERE id = 3 AND EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='icon');
UPDATE categories SET icon = '🧋' WHERE id = 4 AND EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='categories' AND COLUMN_NAME='icon');

-- ============================================
-- SUCCESS MESSAGE
-- ============================================

SELECT 'Schema updates completed successfully!' as status,
       'Tables created: voucher_usage, order_status_history, favorites' as tables_created,
       'Columns added: categories.icon, categories.updated_at, products.is_new, products.is_bestseller, products.is_seasonal' as columns_added;
