# Environment Setup Guide

## Required Tools & Extensions

### 1. Windows Subsystem for Linux (WSL)
Since you're on Windows but need to create ELF binaries (Linux format), you need WSL.

**Install WSL:**
```powershell
# Run in PowerShell as Administrator
wsl --install
```

This installs Ubuntu by default. Restart your computer after installation.

**Verify WSL:**
```powershell
wsl --list --verbose
```

### 2. GCC Compiler (in WSL)
After WSL is installed, open WSL and install build tools:

```bash
# Update package list
sudo apt update

# Install GCC and build essentials
sudo apt install build-essential

# Verify installation
gcc --version
```

### 3. VS Code Extensions

Install these extensions in VS Code:

1. **C/C++** (Microsoft)
   - Extension ID: `ms-vscode.cpptools`
   - Provides IntelliSense, debugging, and code browsing

2. **WSL** (Microsoft)
   - Extension ID: `ms-vscode-remote.remote-wsl`
   - Allows VS Code to work directly with WSL files

3. **Python** (Microsoft) - Already installed
   - Extension ID: `ms-python.python`

**Install via VS Code:**
- Press `Ctrl+Shift+X` to open Extensions
- Search for each extension name
- Click "Install"

**Or install via command:**
```powershell
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode-remote.remote-wsl
```

## Quick Start

### Option 1: Use WSL Directly

1. Open WSL terminal in VS Code:
   - Press `Ctrl+Shift+P`
   - Type "WSL: New Window"
   - Select your WSL distribution

2. Navigate to your project in WSL:
   ```bash
   cd /mnt/c/Users/Nguyen-Bang/VsCode/BCSD-Model-using-GNN-to-enrichen-V-K-/test_binaries
   ```

3. Compile:
   ```bash
   chmod +x compile.sh
   ./compile.sh
   ```

### Option 2: Use WSL from PowerShell

```powershell
# Navigate to test_binaries folder
cd C:\Users\Nguyen-Bang\VsCode\BCSD-Model-using-GNN-to-enrichen-V-K-\test_binaries

# Run compilation in WSL
wsl bash compile.sh
```

## Verification Commands

After compilation:

```bash
# Check binary format (should say "ELF 64-bit" and "not stripped")
wsl file test_binary

# List symbols (should show calculate_sum, check_value, main)
wsl nm test_binary | grep -E "(calculate_sum|check_value|main)"

# Run the binary
wsl ./test_binary
```

## Testing with Your Pipeline

Once the binary is compiled, test it with your angr pipeline:

```powershell
# From PowerShell
python pipeline/angr_disassembly.py test_binaries/test_binary

# Or with visualization
python pipeline/angr_disassembly.py test_binaries/test_binary --visualize main
```

## Troubleshooting

### "gcc: command not found"
Install build-essential in WSL:
```bash
sudo apt update && sudo apt install build-essential
```

### "file not found" when running binary
Make sure you're in WSL or using `wsl` prefix:
```bash
wsl ./test_binary
```

### Python can't read the binary
Make sure the path is accessible from Windows:
```powershell
# Use Windows path
python pipeline/angr_disassembly.py test_binaries/test_binary
```
