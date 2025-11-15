# Changelog

All notable changes to Coffee Shop application will be documented in this file.

## [2.0.0] - 2025-11-15

### 🎉 Major Update - Complete Feature Implementation

#### ✨ New Features

**Cart Management (Hoàn thiện)**
- Full shopping cart UI với item list
- Real-time price calculation
- Quantity adjustment với spinbox
- Individual item removal
- Clear all cart functionality
- Voucher code application
- Order summary với subtotal, discount, delivery fee

**Product Detail Dialog (Hoàn thiện)**
- Detailed product information
- Full customization options:
  - Size selection (S/M/L) với price adjustment
  - Temperature selection (Hot/Cold)
  - Sugar level slider (0-100%)
  - Ice level slider (0-100%)
  - Multiple topping selection
- Real-time price calculation
- Calories display per size
- Quantity selector
- Beautiful responsive layout

**Checkout Flow (Hoàn thiện)**
- Complete checkout dialog
- Order type selection:
  - 🏪 Pickup (chọn cửa hàng)
  - 🚚 Delivery (nhập địa chỉ)
  - 🍽️ Dine-in (nhập số bàn)
- Payment method selection (7 options)
- Order notes
- Order summary preview
- Integration với order creation

**Orders Management (Hoàn thiện)**
- Order history list với detailed info
- Beautiful order card design
- Status tracking với color-coded badges
- Order tracking timeline:
  - Visual timeline với checkmarks
  - Step-by-step status tracking
  - Completed/Pending indicators
- Order detail dialog
- Reorder functionality
- Cancel order (for pending/confirmed orders)
- Refresh orders

**Profile Management (Hoàn thiện)**
- User profile display
- Membership tier display (Bronze/Silver/Gold)
- Loyalty points display
- Points to next tier calculation
- Statistics (total orders, total spent)
- Edit profile (name, phone)
- Change password
- Points history viewer
- Available vouchers viewer

**Menu Enhancements**
- Product cards now open detail dialog for customization
- Quick add removed in favor of full customization

#### 🎨 UI/UX Improvements

- Clean, modern interface
- Coffee-themed color scheme maintained
- Responsive layouts
- Empty states for all lists
- Loading/error handling
- Consistent spacing and styling
- Icon usage throughout
- Better visual hierarchy

#### 🔧 Technical Improvements

- Complete integration between all modules
- Proper signal/slot connections
- State management
- Data refresh mechanisms
- Dialog-based flows
- Error handling
- Input validation

### 📝 Code Organization

**New Files:**
- `ui_generated/cart.py` - Cart UI
- `views/cart_ex.py` - Cart logic với CartItemWidget
- `views/product_detail_dialog.py` - Product customization dialog
- `views/checkout_dialog.py` - Checkout flow
- `views/orders_ex.py` - Orders management với OrderItemWidget, OrderTimelineWidget
- `views/profile_ex.py` - Profile management

**Updated Files:**
- `main.py` - Integration of all widgets, checkout handling
- `views/menu_ex.py` - Product detail dialog integration
- `README.md` - Updated feature list

### 🐛 Bug Fixes

- Fixed cart empty state display
- Fixed price calculation với toppings
- Fixed order status tracking
- Improved error messages

### 🚀 Performance

- Optimized widget rendering
- Lazy loading for dialogs
- Efficient data refresh

---

## [1.0.0] - 2025-11-15

### Initial Release

#### Core Features Implemented

**Authentication System**
- User registration với email/phone
- Login functionality
- Password hashing với SHA-256
- Session management
- OTP infrastructure (backend ready)

**Product Management**
- Product catalog với categories
- Search functionality
- Filter by category, temperature, caffeine
- Product details
- Rating and reviews system (backend)

**Database**
- Complete MySQL schema với 20+ tables
- Sample data
- Relationships và constraints
- Support for all planned features

**UI Framework**
- PyQt6 implementation
- Clean MVC architecture
- Modular structure
- Highland Coffee-inspired design
- Custom stylesheet

**Backend Systems**
- Models: User, Product, Cart, Order, Voucher, etc.
- Controllers: Auth, Menu, Cart, Order, User
- Utilities: Database, Validators, Helpers
- Configuration management

---

## Upcoming Features

### Planned for Future Releases

**Payment Integration**
- MoMo payment gateway
- ZaloPay integration
- ShopeePay integration
- Real payment processing

**Advanced Features**
- AI-based product recommendations
- Real-time GPS tracking
- Push notifications
- QR code table ordering
- Image upload for products
- Advanced analytics

**UI Enhancements**
- Product images support
- Custom icons
- Animations
- Loading states
- Dark mode (optional)

**Social Features**
- Review submission UI
- Photo reviews
- Social sharing

---

## Notes

- Version 2.0.0 focuses on completing core customer-facing features
- All major user flows are now functional
- Ready for testing and feedback
- Database schema supports all implemented features
