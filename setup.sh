check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Step 1: Set up virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv  # Ganti dengan 'python3' atau 'python' sesuai dengan kebutuhan
check_error "Failed to create virtual environment."

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
# Gunakan command untuk mengaktifkan environment sesuai dengan sistem yang digunakan
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Unable to find virtual environment activation script."
    exit 1
fi
check_error "Failed to activate virtual environment."

# Step 3: Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
check_error "Failed to install dependencies."

echo "Setup completed successfully."
exit 0  # Hapus exit 1 yang tidak diperlukan
