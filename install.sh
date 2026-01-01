#!/bin/bash

# ==============================================================================
# AI Coding Assistant - Auto Installer (Linux & Termux)
# ==============================================================================

echo "🚀 Starting AI Coding Assistant Auto-Installer..."

# 1. تحديد نوع النظام
if [ -d "/data/data/com.termux" ]; then
    OS="Termux"
    INSTALL_CMD="pkg install -y"
    UPDATE_CMD="pkg update -y"
else
    OS="Linux"
    INSTALL_CMD="sudo apt-get install -y"
    UPDATE_CMD="sudo apt-get update -y"
fi

echo "📍 Detected OS: $OS"

# 2. تحديث المستودعات وتثبيت الأدوات الأساسية
echo "🔄 Updating repositories and installing base tools (wget, git, python)..."
$UPDATE_CMD
$INSTALL_CMD wget git python3 python3-pip

# 3. تثبيت المكتبات البرمجية باستخدام pip
echo "📦 Installing Python libraries (transformers, datasets, gradio, etc.)..."
# استخدام --no-cache-dir لتوفير المساحة في Termux
pip3 install --upgrade pip
pip3 install transformers datasets peft bitsandbytes accelerate gradio torch --no-cache-dir

# 4. إعداد بيئة العمل
echo "📂 Setting up project directories..."
mkdir -p raw_data
mkdir -p models

# 5. رسالة النجاح
echo "===================================================="
echo "✅ Installation Complete!"
echo "===================================================="
echo "To start the GUI, run: python3 app_gui.py"
echo "To start training, run: python3 zip_training_script.py"
echo "===================================================="

if [ "$OS" == "Termux" ]; then
    echo "💡 Note for Termux: Make sure you have enough storage (at least 2GB) for the models."
fi
