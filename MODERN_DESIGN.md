# ☕ Coffee Shop - Modern Design Theme

## 🎨 Design Philosophy: "Warm Artisan Coffee House"

Thiết kế mới của Coffee Shop App được lấy cảm hứng từ không gian quán cafe thủ công (artisan coffee house) với sự kết hợp hài hòa giữa sự ấm áp, sang trọng và hiện đại.

---

## 🌟 Aesthetic Direction

### Core Concept
- **Warmth & Comfort**: Tông màu ấm của espresso, caramel, cream tạo cảm giác thư giãn
- **Premium & Refined**: Typography và spacing được chọn lọc kỹ lưỡng
- **Organic & Natural**: Gradients mượt mà, border-radius tròn trịa
- **Tactile Experience**: Shadows và hover effects tạo cảm giác chạm được

### Design Principles
1. **Sophistication over Flash**: Tinh tế thay vì rực rỡ
2. **Consistency is Key**: Mọi element đều theo một hệ thống nhất quán
3. **User Delight**: Micro-interactions làm trải nghiệm thú vị hơn
4. **Accessibility**: Dễ đọc, dễ nhìn, dễ sử dụng

---

## 🎨 Color Palette

### Primary Colors
```css
--espresso: #2D1B10       /* Text chính, backgrounds tối */
--dark-roast: #3E2723     /* Hover states, secondary text */
--coffee-bean: #4E342E    /* Buttons, accents */
```

### Accent Colors
```css
--caramel: #D4A574        /* Gradient stops, highlights */
--latte: #E8D5C4          /* Borders, subtle backgrounds */
--cream: #F5F0EB          /* Card backgrounds */
--foam: #FDFBF9           /* Main background */
```

### Metallic Accents
```css
--accent-gold: #C9A961    /* Primary actions, active states */
--accent-copper: #B87333  /* Hover effects, secondary accents */
```

---

## 📝 Typography System

### Font Stack
```css
'Segoe UI', -apple-system, BlinkMacSystemFont,
'SF Pro Display', system-ui, sans-serif
```

### Type Scale
- **Display**: 32px, weight 700 (Titles)
- **Heading**: 20-28px, weight 600-700
- **Body**: 15px, weight 400
- **Caption**: 13px, weight 400

### Features
- Letter spacing: 0.3px - 1px cho headings
- Line height: 1.6 cho body text
- Italic cho placeholders

---

## 🧩 Component Designs

### 1. Login/Register Screens

**Layout Strategy**:
- Horizontal spacers để content luôn căn giữa
- Max-width: 450px cho content widget
- Min-height fixed cho buttons/inputs

**Visual Details**:
- Primary buttons với gold gradient (#C9A961 → #D4A574)
- Input fields: soft white background, 2px borders
- Placeholder text: italic, #A08B7D

### 2. Sidebar (Glassmorphism Effect)

**Background**:
```css
qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 rgba(62, 39, 35, 0.95),
    stop:1 rgba(45, 27, 16, 0.98))
```

**Features**:
- Semi-transparent dark background
- Gold accent border (rgba 20% opacity)
- Smooth hover transitions
- Active state with gold gradient

### 3. Product Cards

**Dimensions**:
- Max-width: 280px
- Min-height: 400px
- Image: 244x244px with 20px border-radius

**Styling**:
- Card background: subtle gradient (white → cream)
- Hover effect: border color changes to gold
- Image background: tricolor gradient
- Favorite button: glassmorphism overlay

**Typography**:
- Product name: 17px, weight 700, #2D1B10
- Price: 20px, weight 800, gold color
- Rating: 13px with star emoji

### 4. Buttons

#### Primary (Login/Checkout)
```css
background: gradient(#C9A961 → #D4A574 → #B87333)
color: #2D1B10
min-height: 56px
border-radius: 16px
```

#### Secondary (Add to Cart)
```css
background: gradient(#C9A961 → #D4A574)
min-height: 48px
border-radius: 12px
```

#### Logout Button
```css
background: transparent with subtle gradient
border: 1px solid gold (25% opacity)
color: cream
```

### 5. Input Fields

**Default State**:
- Background: #FFFFFF
- Border: 2px #E8D5C4
- Padding: 14px 20px
- Height: 52-58px

**Focus State**:
- Border color → #C9A961
- Background → #FDFBF9

**Search Box** (Special):
- Border-radius: 28px (pill shape)
- Gradient background
- Left padding: 48px (icon space)

### 6. Tab Widget

**Design**:
- No border around pane
- Top border: 2px #E8D5C4
- Selected tab: gold gradient background
- Hover: subtle background (15% opacity)

**Spacing**:
- Padding: 16px 32px
- Margin-right: 8px
- Border-radius: 14px

### 7. Scrollbars

**Minimal Design**:
- Width/Height: 12px
- Handle: gold gradient
- Transparent track
- Smooth hover animation

---

## ✨ Visual Effects

### Gradients

**Main Background**:
```css
qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #FDFBF9,
    stop:0.5 #F5F0EB,
    stop:1 #E8D5C4)
```

**Primary Buttons**:
```css
qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #C9A961,
    stop:0.5 #D4A574,
    stop:1 #B87333)
```

**Card Backgrounds**:
```css
qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 #FFFFFF,
    stop:1 #F5F0EB)
```

### Shadows

**Product Cards** (on hover):
```css
box-shadow: 0 4px 12px rgba(111, 78, 55, 0.15)
```

**Buttons** (active state):
```css
box-shadow: 0 4px 12px rgba(201, 169, 97, 0.3)
```

### Border Radius

- Cards: 16-20px
- Buttons: 12-16px
- Inputs: 14px
- Search box: 28px (pill)
- Checkboxes: 8px
- Radio buttons: 12px (circle)

---

## 🎭 Interactive States

### Hover Effects

**Buttons**:
- Lighter gradient
- Subtle transform (optional)

**Cards**:
- Border color → gold
- Background → lighter shade
- Shadow appears

**Sidebar Items**:
- Background: gold gradient (15% opacity)
- Text color → gold

### Active/Selected States

**Tabs**:
- Background: full gold gradient
- Text color: espresso (#2D1B10)
- Font weight: 700

**Sidebar**:
- Background: solid gold gradient
- Text color: espresso
- Font weight: 600

### Focus States

**Inputs**:
- Border: 2px gold (#C9A961)
- Background: slightly lighter

---

## 📱 Layout Guidelines

### Spacing System

**Margins**:
- Components: 12-24px
- Sections: 20-32px

**Padding**:
- Small: 8-12px
- Medium: 14-18px
- Large: 20-32px

**Gaps**:
- Between elements: 12-14px
- Between sections: 16-20px

### Grid System

**Product Grid**:
- Max columns: 3
- Gap: auto (spacing by cards)
- Responsive: collapses on smaller screens

### Container Constraints

**Login/Register**:
- Window max-width: 500px
- Content max-width: 450px
- Centered with horizontal spacers

**Main Window**:
- Sidebar: max-width 250px
- Content area: flexible
- Min window: 1200x800px

---

## 🔧 Implementation Details

### File Structure

```
resources/styles/
├── modern_style.qss       # Modern theme (NEW)
└── style.qss             # Classic theme (OLD)

utils/
└── config.py             # Theme configuration
```

### Configuration

**utils/config.py**:
```python
USE_MODERN_THEME = True   # Set to False for classic theme
MODERN_STYLESHEET = STYLES_DIR / 'modern_style.qss'
CLASSIC_STYLESHEET = STYLES_DIR / 'style.qss'
```

### Switching Themes

To switch between themes, edit `utils/config.py`:

```python
# For Modern Theme
USE_MODERN_THEME = True

# For Classic Theme
USE_MODERN_THEME = False
```

Then restart the application.

---

## 🎯 Design Highlights

### What Makes This Design Special

1. **Cohesive Color Story**: Mọi màu sắc đều lấy cảm hứng từ cafe
2. **Premium Feel**: Gradients, shadows, và typography tạo cảm giác cao cấp
3. **Warm & Inviting**: Tông màu ấm tạo không gian thân thiện
4. **Attention to Detail**: Mọi pixel đều được tính toán kỹ
5. **Smooth Transitions**: Hover effects mượt mà, tự nhiên

### Differentiation Points

- **No Generic Blues/Purples**: Hoàn toàn tránh cliché colors
- **Coffee-Inspired Palette**: Unique và memorable
- **Glassmorphism Sidebar**: Modern trend được áp dụng khéo léo
- **Asymmetric Layouts**: Phá vỡ sự đơn điệu
- **Typography Hierarchy**: Rõ ràng và đẹp mắt

---

## 📊 Before & After Comparison

### Classic Theme
- ❌ Generic Inter font
- ❌ Standard blue/purple colors
- ❌ Flat, predictable layouts
- ❌ Minimal visual interest

### Modern Theme
- ✅ System fonts với fallbacks đẹp
- ✅ Unique coffee-inspired palette
- ✅ Depth với gradients & shadows
- ✅ Premium, artisan feeling

---

## 🚀 Performance Considerations

- **CSS-only animations**: Không dùng JavaScript
- **Gradients**: Native Qt gradients, không cần images
- **Minimal assets**: Chỉ dùng CSS và colors
- **Fast rendering**: Tối ưu cho PyQt6

---

## 🎨 Customization Guide

### Changing Primary Color

Tìm và thay thế các values:
- `#C9A961` (accent-gold)
- `#D4A574` (caramel)
- `#B87333` (accent-copper)

### Adjusting Warmth

Để tăng/giảm độ "ấm":
- Tăng: Dùng nhiều `#D4A574`, `#E8D5C4`
- Giảm: Dùng nhiều `#FFFFFF`, `#F5F0EB`

### Border Radius

Tìm `border-radius` và điều chỉnh:
- Mềm mại hơn: Tăng lên (20-24px)
- Sắc nét hơn: Giảm xuống (8-10px)

---

## 📝 Credits & Inspiration

**Design Inspiration**:
- Artisan coffee shops
- Premium coffee brands (Blue Bottle, Stumptown)
- Material Design 3
- iOS design language

**Color Palette**:
- Inspired by coffee beans, espresso, latte art
- Warm earth tones with metallic accents

**Typography**:
- System fonts for performance
- Clean, modern sans-serif family

---

## 🎉 Conclusion

Thiết kế mới của Coffee Shop App không chỉ đẹp mắt mà còn:
- **Functional**: Dễ sử dụng, rõ ràng
- **Memorable**: Độc đáo, khó quên
- **Cohesive**: Thống nhất từ đầu đến cuối
- **Premium**: Cảm giác cao cấp, chuyên nghiệp

Enjoy your new beautiful coffee shop experience! ☕✨
