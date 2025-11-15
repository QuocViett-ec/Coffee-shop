#!/bin/bash
# Setup Admin Panel Database
# This script creates admin tables and default admin account

echo "=================================="
echo "Coffee Shop Admin Panel Setup"
echo "=================================="
echo ""

# Check if MySQL is running
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL client not found. Please install MySQL first."
    exit 1
fi

echo "This script will create admin tables in the coffee_shop database."
echo ""
read -p "MySQL root password: " -s MYSQL_PASSWORD
echo ""

# Import admin schema
echo "Creating admin tables..."
mysql -u root -p"$MYSQL_PASSWORD" coffee_shop < database/admin_schema.sql

if [ $? -eq 0 ]; then
    echo "✅ Admin tables created successfully!"
    echo ""
    echo "=================================="
    echo "Default Admin Account:"
    echo "=================================="
    echo "Username: admin"
    echo "Password: admin123"
    echo ""
    echo "⚠️  Please change the default password after first login!"
    echo ""
    echo "To start admin panel, run:"
    echo "  python admin.py"
else
    echo "❌ Failed to create admin tables"
    exit 1
fi
