-- Coffee Shop Database Schema Updates
-- Updates for new features (Favorites, Admin Management, etc.)
-- Run this after schema.sql

USE coffee_shop;

-- ============================================
-- 1. UPDATE CATEGORIES TABLE
-- ============================================
-- Add 'icon' field for emoji icons (used in admin panel)
ALTER TABLE categories
ADD COLUMN IF NOT EXISTS icon VARCHAR(10) DEFAULT '☕' AFTER description;

-- Update updated_at field
ALTER TABLE categories
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;

-- ============================================
-- 2. UPDATE PRODUCTS TABLE
-- ============================================
-- Add fields for product tags (used in admin product management)
ALTER TABLE products
ADD COLUMN IF NOT EXISTS is_new BOOLEAN DEFAULT FALSE AFTER is_featured;

ALTER TABLE products
ADD COLUMN IF NOT EXISTS is_bestseller BOOLEAN DEFAULT FALSE AFTER is_new;

ALTER TABLE products
ADD COLUMN IF NOT EXISTS is_seasonal BOOLEAN DEFAULT FALSE AFTER is_bestseller;

-- ============================================
-- 3. CREATE VOUCHER_USAGE TABLE
-- ============================================
-- Track voucher usage (required for admin voucher management)
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
-- Track order status changes (required for admin order management)
CREATE TABLE IF NOT EXISTS order_status_history (
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
) ENGINE=InnoDB;

-- ============================================
-- 5. RENAME user_favorites to favorites
-- ============================================
-- Rename table to match code (if needed)
-- Note: Only run if table is named 'user_favorites' instead of 'favorites'
-- RENAME TABLE user_favorites TO favorites;

-- Or create favorites table if it doesn't exist
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
-- Update existing products with new fields
UPDATE products SET is_new = TRUE WHERE id IN (1, 2, 3) LIMIT 3;
UPDATE products SET is_bestseller = TRUE WHERE id IN (1, 4, 5) LIMIT 3;
UPDATE products SET is_seasonal = FALSE;

-- Update categories with icons
UPDATE categories SET icon = '☕' WHERE id = 1;
UPDATE categories SET icon = '🥤' WHERE id = 2;
UPDATE categories SET icon = '🍰' WHERE id = 3;
UPDATE categories SET icon = '🧋' WHERE id = 4;

-- ============================================
-- VERIFICATION QUERIES
-- ============================================
-- Run these to verify the updates

-- Check categories structure
-- SHOW COLUMNS FROM categories;

-- Check products structure
-- SHOW COLUMNS FROM products;

-- Check new tables exist
-- SHOW TABLES LIKE '%voucher_usage%';
-- SHOW TABLES LIKE '%order_status_history%';
-- SHOW TABLES LIKE 'favorites';

-- Check counts
-- SELECT COUNT(*) as total_products FROM products;
-- SELECT COUNT(*) as total_categories FROM categories;
-- SELECT COUNT(*) as total_vouchers FROM vouchers;

SELECT 'Schema updates completed successfully!' as status;
