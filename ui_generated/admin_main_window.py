"""
Auto-generated UI file for Admin Main Window
"""
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_AdminMainWindow(object):
    def setupUi(self, AdminMainWindow):
        AdminMainWindow.setObjectName("AdminMainWindow")
        AdminMainWindow.resize(1400, 900)

        self.centralWidget = QtWidgets.QWidget(AdminMainWindow)
        self.centralWidget.setObjectName("centralWidget")

        self.mainLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        # Sidebar
        self.sidebarWidget = QtWidgets.QWidget(self.centralWidget)
        self.sidebarWidget.setMaximumWidth(250)
        self.sidebarWidget.setObjectName("sidebarWidget")

        self.sidebarLayout = QtWidgets.QVBoxLayout(self.sidebarWidget)

        # Logo
        self.logoLabel = QtWidgets.QLabel("☕ Admin Panel")
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.logoLabel.setFont(font)
        self.logoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logoLabel.setStyleSheet("color: #c7a17a; padding: 20px 0;")
        self.sidebarLayout.addWidget(self.logoLabel)

        # Admin info
        self.adminInfoWidget = QtWidgets.QFrame()
        self.adminInfoWidget.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.adminInfoWidget.setObjectName("adminInfoWidget")

        self.adminInfoLayout = QtWidgets.QVBoxLayout(self.adminInfoWidget)

        self.adminNameLabel = QtWidgets.QLabel("Admin")
        font = QtGui.QFont()
        font.setBold(True)
        self.adminNameLabel.setFont(font)
        self.adminInfoLayout.addWidget(self.adminNameLabel)

        self.adminRoleLabel = QtWidgets.QLabel("👑 Super Admin")
        self.adminInfoLayout.addWidget(self.adminRoleLabel)

        self.sidebarLayout.addWidget(self.adminInfoWidget)

        # Navigation buttons
        self.dashboardButton = QtWidgets.QPushButton("📊 Dashboard")
        self.dashboardButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.dashboardButton)

        self.ordersButton = QtWidgets.QPushButton("📦 Quản lý đơn hàng")
        self.ordersButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.ordersButton)

        self.productsButton = QtWidgets.QPushButton("☕ Quản lý sản phẩm")
        self.productsButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.productsButton)

        self.usersButton = QtWidgets.QPushButton("👥 Quản lý khách hàng")
        self.usersButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.usersButton)

        self.categoriesButton = QtWidgets.QPushButton("📂 Quản lý danh mục")
        self.categoriesButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.categoriesButton)

        self.vouchersButton = QtWidgets.QPushButton("🎫 Quản lý voucher")
        self.vouchersButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.vouchersButton)

        self.reportsButton = QtWidgets.QPushButton("📈 Báo cáo")
        self.reportsButton.setMinimumHeight(45)
        self.sidebarLayout.addWidget(self.reportsButton)

        # Spacer
        spacerItem = QtWidgets.QSpacerItem(20, 40,
                                           QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.sidebarLayout.addItem(spacerItem)

        # Logout button
        self.logoutButton = QtWidgets.QPushButton("🚪 Đăng xuất")
        self.logoutButton.setMinimumHeight(40)
        self.sidebarLayout.addWidget(self.logoutButton)

        self.mainLayout.addWidget(self.sidebarWidget)

        # Content area
        self.contentStackedWidget = QtWidgets.QStackedWidget(self.centralWidget)
        self.contentStackedWidget.setObjectName("contentStackedWidget")
        self.mainLayout.addWidget(self.contentStackedWidget)

        AdminMainWindow.setCentralWidget(self.centralWidget)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(AdminMainWindow)
        AdminMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AdminMainWindow)
        QtCore.QMetaObject.connectSlotsByName(AdminMainWindow)

    def retranslateUi(self, AdminMainWindow):
        _translate = QtCore.QCoreApplication.translate
        AdminMainWindow.setWindowTitle(_translate("AdminMainWindow", "Coffee Shop Admin"))
