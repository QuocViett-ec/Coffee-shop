"""
Menu Widget - Extended Logic
Display products with filtering and search
"""
from PyQt6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from ui_generated.menu import Ui_MenuWidget
from controllers.menu_controller import MenuController
from controllers.cart_controller import CartController
from controllers.auth_controller import AuthController
from utils.validators import format_currency


class ProductCard(QFrame):
    """Product card widget"""

    add_to_cart_clicked = pyqtSignal(dict)
    product_clicked = pyqtSignal(dict)

    def __init__(self, product_data, parent=None):
        super().__init__(parent)
        self.product_data = product_data
        self.setup_ui()

    def setup_ui(self):
        """Setup product card UI"""
        self.setFrameShape(QFrame.Shape.Box)
        self.setMaximumWidth(250)
        self.setMinimumHeight(320)

        layout = QVBoxLayout(self)

        # Product image (placeholder)
        image_label = QLabel()
        image_label.setFixedSize(220, 220)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setText("📷")  # Placeholder
        image_label.setStyleSheet("background-color: #f0f0f0; border-radius: 8px;")
        layout.addWidget(image_label)

        # Product name
        name_label = QLabel(self.product_data['name'])
        name_label.setWordWrap(True)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(name_label)

        # Price
        price = format_currency(self.product_data['base_price'])
        price_label = QLabel(price)
        price_label.setStyleSheet("color: #d4691e; font-size: 16px; font-weight: bold;")
        layout.addWidget(price_label)

        # Rating
        rating = self.product_data.get('rating', 0)
        total_reviews = self.product_data.get('total_reviews', 0)
        rating_label = QLabel(f"⭐ {rating:.1f} ({total_reviews} đánh giá)")
        rating_label.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(rating_label)

        # Add to cart button
        add_btn = QPushButton("Thêm vào giỏ")
        add_btn.clicked.connect(lambda: self.add_to_cart_clicked.emit(self.product_data))
        layout.addWidget(add_btn)

        # Make card clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        """Handle card click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.product_clicked.emit(self.product_data)


class MenuWidget(QWidget, Ui_MenuWidget):
    """Menu widget with product display and filtering"""

    cart_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.menu_controller = MenuController()
        self.cart_controller = CartController()
        self.auth_controller = AuthController()

        # Connect signals
        self.searchLineEdit.textChanged.connect(self.handle_search)
        self.hotCheckBox.stateChanged.connect(self.apply_filters)
        self.coldCheckBox.stateChanged.connect(self.apply_filters)
        self.caffeineCheckBox.stateChanged.connect(self.apply_filters)

        # Load categories and products
        self.load_categories()
        self.load_products()

    def load_categories(self):
        """Load product categories as tabs"""
        categories = self.menu_controller.get_categories()

        # Clear existing tabs
        self.categoryTabWidget.clear()

        # Add "All" tab
        all_tab = QWidget()
        self.categoryTabWidget.addTab(all_tab, "Tất cả")

        # Add category tabs
        for category in categories:
            tab = QWidget()
            self.categoryTabWidget.addTab(tab, category['name'])

        # Connect tab change signal
        self.categoryTabWidget.currentChanged.connect(self.handle_category_change)

    def load_products(self, category_id=None):
        """Load and display products"""
        products = self.menu_controller.get_products_by_category(category_id)
        self.display_products(products)

    def display_products(self, products):
        """Display products in grid layout"""
        # Clear existing products
        while self.productsGridLayout.count():
            item = self.productsGridLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add products to grid
        row = 0
        col = 0
        max_cols = 3

        for product in products:
            card = ProductCard(product)
            card.add_to_cart_clicked.connect(self.handle_add_to_cart)
            card.product_clicked.connect(self.handle_product_click)

            self.productsGridLayout.addWidget(card, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Add stretch to fill remaining space
        self.productsGridLayout.setRowStretch(row + 1, 1)

    def handle_category_change(self, index):
        """Handle category tab change"""
        if index == 0:
            # All products
            self.load_products()
        else:
            # Get category ID
            categories = self.menu_controller.get_categories()
            if 0 <= index - 1 < len(categories):
                category_id = categories[index - 1]['id']
                self.load_products(category_id)

    def handle_search(self, query):
        """Handle product search"""
        if not query.strip():
            # If search is empty, reload current category
            current_index = self.categoryTabWidget.currentIndex()
            self.handle_category_change(current_index)
            return

        # Search products
        products = self.menu_controller.search_products(query)
        self.display_products(products)

    def apply_filters(self):
        """Apply product filters"""
        # Get current category
        current_index = self.categoryTabWidget.currentIndex()
        category_id = None

        if current_index > 0:
            categories = self.menu_controller.get_categories()
            if 0 <= current_index - 1 < len(categories):
                category_id = categories[current_index - 1]['id']

        # Get filter values
        temperature = None
        if self.hotCheckBox.isChecked():
            temperature = 'hot'
        elif self.coldCheckBox.isChecked():
            temperature = 'cold'

        is_caffeine_free = self.caffeineCheckBox.isChecked() if self.caffeineCheckBox.isChecked() else None

        # Filter products
        products = self.menu_controller.filter_products(
            category_id=category_id,
            temperature=temperature,
            is_caffeine_free=is_caffeine_free
        )

        self.display_products(products)

    def handle_add_to_cart(self, product):
        """Handle add to cart button click - show detail dialog"""
        # Show product detail dialog for customization
        self.handle_product_click(product)

    def handle_product_click(self, product):
        """Handle product card click - show detail dialog"""
        from views.product_detail_dialog import ProductDetailDialog

        dialog = ProductDetailDialog(product['id'], self)
        dialog.product_added.connect(self.cart_updated.emit)
        dialog.exec()
