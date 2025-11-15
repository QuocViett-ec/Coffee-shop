@echo off
REM Quick fix for favorites table error
REM This creates the missing favorites table

echo =========================================
echo Coffee Shop - Fix Favorites Table
echo =========================================
echo.

REM Check if MySQL is accessible
where mysql >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MySQL not found in PATH!
    echo.
    echo Please either:
    echo 1. Add MySQL to PATH (see SETUP_WINDOWS.md)
    echo 2. Use MySQL Workbench to run: fix_favorites.sql
    echo 3. Use phpMyAdmin to run: fix_favorites.sql
    echo.
    pause
    exit /b 1
)

set /p MYSQL_PASSWORD="Enter MySQL root password (press Enter if no password): "
echo.

echo Creating favorites table...
mysql -u root -p%MYSQL_PASSWORD% < fix_favorites.sql

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Favorites table created!
    echo.
    echo You can now run the app:
    echo   python main.py
) else (
    echo.
    echo [ERROR] Failed to create table
    echo Please check your MySQL password.
)

echo.
pause
