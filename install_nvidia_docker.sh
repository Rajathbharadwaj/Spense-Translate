#!/bin/bash

echo "📦 Installing NVIDIA Container Toolkit for Arch/Garuda Linux..."

# For Arch-based systems (including Garuda)
if command -v yay &> /dev/null; then
    echo "Installing with yay..."
    yay -S nvidia-container-toolkit
elif command -v paru &> /dev/null; then
    echo "Installing with paru..."
    paru -S nvidia-container-toolkit
else
    echo "Installing from AUR manually..."
    git clone https://aur.archlinux.org/nvidia-container-toolkit.git
    cd nvidia-container-toolkit
    makepkg -si
    cd ..
    rm -rf nvidia-container-toolkit
fi

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "✅ NVIDIA Container Toolkit installed!"
echo "🔧 Docker configured for GPU support"
echo ""
echo "Now you can run: ./run_ngc.sh"