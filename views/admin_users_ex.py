"""Admin Users Management"""
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from controllers.admin_user_controller import AdminUserController
from controllers.admin_controller import AdminController
from utils.validators import format_currency
from datetime import datetime

class AdminUsersWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_controller = AdminUserController()
        self.admin_controller = AdminController()
        self.setup_ui()
        self.load_users()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("<h2>👥 Quản lý khách hàng</h2>"))
        header.addStretch()
        refresh_btn = QPushButton("🔄 Làm mới")
        refresh_btn.clicked.connect(self.load_users)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Tìm kiếm:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Tên, email, SĐT...")
        self.search_edit.textChanged.connect(self.load_users)
        filter_layout.addWidget(self.search_edit)

        filter_layout.addWidget(QLabel("Tier:"))
        self.tier_combo = QComboBox()
        self.tier_combo.addItem("Tất cả", None)
        for tier in ['Bronze', 'Silver', 'Gold']:
            self.tier_combo.addItem(tier, tier)
        self.tier_combo.currentIndexChanged.connect(self.load_users)
        filter_layout.addWidget(self.tier_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Tên", "Email", "SĐT", "Tier", "Điểm", "Tổng chi", "Thao tác"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def load_users(self):
        search = self.search_edit.text().strip()
        tier = self.tier_combo.currentData()
        users = self.user_controller.get_all_users(
            search=search if search else None,
            tier=tier
        )
        self.display_users(users)

    def display_users(self, users):
        self.table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.table.setItem(row, 0, QTableWidgetItem(str(user['id'])))
            self.table.setItem(row, 1, QTableWidgetItem(user['full_name']))
            self.table.setItem(row, 2, QTableWidgetItem(user.get('email', 'N/A')))
            self.table.setItem(row, 3, QTableWidgetItem(user.get('phone', 'N/A')))
            
            tier_icons = {'Bronze': '🥉', 'Silver': '🥈', 'Gold': '🥇'}
            tier_text = f"{tier_icons.get(user['membership_tier'], '')} {user['membership_tier']}"
            self.table.setItem(row, 4, QTableWidgetItem(tier_text))
            
            self.table.setItem(row, 5, QTableWidgetItem(str(user.get('loyalty_points', 0))))
            self.table.setItem(row, 6, QTableWidgetItem(format_currency(user.get('total_spent', 0))))

            # Actions
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(5, 2, 5, 2)

            tier_btn = QPushButton("🏆")
            tier_btn.setToolTip("Đổi tier")
            tier_btn.clicked.connect(lambda checked, u=user: self.change_tier(u))
            action_layout.addWidget(tier_btn)

            points_btn = QPushButton("💰")
            points_btn.setToolTip("Điểm")
            points_btn.clicked.connect(lambda checked, u=user: self.adjust_points(u))
            action_layout.addWidget(points_btn)

            self.table.setCellWidget(row, 7, action_widget)

        self.table.resizeColumnsToContents()

    def change_tier(self, user):
        tiers = ['Bronze', 'Silver', 'Gold']
        tier, ok = QInputDialog.getItem(self, "Đổi Tier", f"Chọn tier cho {user['full_name']}:", tiers, tiers.index(user['membership_tier']), False)
        if ok:
            admin_id = self.admin_controller.get_current_admin_id()
            success, msg = self.user_controller.update_user_tier(user['id'], tier, admin_id)
            if success:
                QMessageBox.information(self, "Thành công", msg)
                self.load_users()
            else:
                QMessageBox.warning(self, "Lỗi", msg)

    def adjust_points(self, user):
        points, ok = QInputDialog.getInt(self, "Điều chỉnh điểm", f"Nhập số điểm cộng/trừ cho {user['full_name']}:\n(Số âm để trừ)", 0, -100000, 100000, 1)
        if ok and points != 0:
            admin_id = self.admin_controller.get_current_admin_id()
            success, msg = self.user_controller.update_loyalty_points(user['id'], points, admin_id)
            if success:
                QMessageBox.information(self, "Thành công", msg)
                self.load_users()
            else:
                QMessageBox.warning(self, "Lỗi", msg)

    def refresh(self):
        self.load_users()
