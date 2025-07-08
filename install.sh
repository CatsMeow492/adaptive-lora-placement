#!/bin/bash
# Installation script for Adaptive LoRA research project

echo "🔧 Setting up Adaptive LoRA research environment..."

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv adaptive-lora-env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source adaptive-lora-env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e ".[dev]"

echo "✅ Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  source adaptive-lora-env/bin/activate"
echo ""
echo "To get started, run:"
echo "  python scripts/train_baseline.py --help" 