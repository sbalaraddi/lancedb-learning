# Python & LanceDB Installation Guide

## Step 1: Install Python

### Method 1: Official Python Installer (Recommended)

1. **Download Python:**
   - Visit: https://www.python.org/downloads/
   - Click the yellow "Download Python 3.12.x" button
   - Save the installer

2. **Run the Installer:**
   - Double-click the downloaded file (e.g., `python-3.12.x-amd64.exe`)
   - ⚠️ **CRITICAL:** Check the box "Add Python to PATH" at the bottom
   - Click "Install Now"
   - Wait for installation to complete
   - Click "Close"

3. **Verify Installation:**
   - Open a NEW PowerShell window (important - close old ones)
   - Run these commands:
   ```powershell
   python --version
   # Should show: Python 3.12.x
   
   pip --version
   # Should show: pip 24.x.x from ...
   ```

### Method 2: Microsoft Store (Easier)

1. Open Microsoft Store app
2. Search for "Python 3.12"
3. Click "Get" or "Install"
4. Wait for installation
5. Open a NEW PowerShell window
6. Verify with `python --version`

### Method 3: Winget (Command Line)

```powershell
winget install Python.Python.3.12
```

---

## Step 2: Install LanceDB

Once Python is installed, open a NEW PowerShell window and run:

```powershell
# Navigate to your project folder
cd C:\Learning\LanceDB

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install LanceDB and dependencies
pip install lancedb numpy pandas

# For the full tutorial (includes optional packages)
pip install -r requirements.txt
```

---

## Step 3: Verify Everything Works

```powershell
# Quick test
python -c "import lancedb; print('LanceDB installed successfully!')"
```

---

## Step 4: Start Learning!

```powershell
# Run the first tutorial
python 01_basic_operations.py

# Continue with others
python 02_vector_search.py
python 03_filtering_queries.py
# ... and so on
```

---

## Troubleshooting

### Issue: "python is not recognized"
**Solution:** Python is not in PATH
- Reinstall Python and CHECK "Add Python to PATH"
- OR manually add to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit System Environment Variables
  3. Add Python paths (e.g., `C:\Python312\` and `C:\Python312\Scripts\`)
  4. Restart PowerShell

### Issue: "pip is not recognized"
**Solution:** Use python -m pip instead
```powershell
python -m pip install lancedb
```

### Issue: Permission denied
**Solution:** Run as administrator or install for user only
```powershell
pip install --user lancedb
```

### Issue: SSL/Certificate errors
**Solution:** Upgrade pip or use trusted host
```powershell
python -m pip install --upgrade pip
# OR
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org lancedb
```

---

## Quick Start After Installation

1. ✅ Python installed
2. ✅ pip working
3. ✅ LanceDB installed
4. 🚀 Run: `python 01_basic_operations.py`

---

## Need Help?

- Python docs: https://docs.python.org/3/
- LanceDB docs: https://lancedb.github.io/lancedb/
- Check `LANCEDB_REFERENCE.md` for API reference

---

**Ready to start? Install Python first, then come back here!** 🎯
