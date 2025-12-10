# GitHub Setup Guide - Upload LanceDB Learning Project

## Prerequisites
- GitHub account (create one at https://github.com if you don't have one)
- Git installed on your computer

---

## Step 1: Install Git (if not already installed)

### Check if Git is installed:
```powershell
git --version
```

### If not installed, download and install Git:
1. Visit: https://git-scm.com/download/win
2. Download the installer
3. Run the installer with default settings
4. Restart PowerShell after installation

---

## Step 2: Configure Git for This Project (Repository-Level Config)

**IMPORTANT:** Since you use Git for office work with corporate credentials, we'll set up project-specific Git configuration instead of global configuration. This keeps your corporate Git settings unchanged.

```powershell
# Navigate to your project directory
cd C:\Learning\LanceDB

# FIRST: Initialize Git repository (required before configuration)
git init

# Set your PERSONAL name for THIS PROJECT ONLY (not global)
git config user.name "Santosh Balaraddi"

# Set your PERSONAL GitHub email for THIS PROJECT ONLY (not global)
git config user.email "sbalaraddi@yahoo.com"

# Verify project-specific configuration
git config --local --list

# Check what's configured (should show your personal info)
git config user.name
git config user.email
```

**Why use `git config` without `--global`?**
- ✅ Only affects THIS project (`C:\Learning\LanceDB`)
- ✅ Your corporate Git settings remain unchanged
- ✅ Office repositories continue using corporate credentials
- ✅ Personal projects use personal credentials

**Your global (corporate) settings are preserved:**
```powershell
# View your global (office) settings - DO NOT CHANGE
git config --global user.name
git config --global user.email
```

---

## Step 3: Create a New Repository on GitHub

1. **Go to GitHub:** https://github.com
2. **Sign in** to your account
3. **Click the "+" icon** in the top-right corner
4. **Select "New repository"**
5. **Fill in repository details:**
   - **Repository name:** `lancedb-learning` (or any name you prefer)
   - **Description:** "Complete LanceDB tutorial project with 9 hands-on modules covering vector search, RAG systems, and semantic search"
   - **Visibility:** Choose "Public" or "Private"
   - **DO NOT** check "Initialize with README" (we already have files)
   - **DO NOT** add .gitignore or license yet
6. **Click "Create repository"**
## Step 4: Verify Git Initialization and Status

```powershell
# You already initialized Git in Step 2, verify it worked
cd C:\Learning\LanceDB

# Check status (should show untracked files)
git status

# Verify your personal credentials are set
git config user.name
git config user.email
```

**Expected output:**
- `user.name`: Santosh Balaraddi
- `user.email`: sbalaraddi@yahoo.com
# Check status
git status
```

---

## Step 5: Create .gitignore File

This prevents uploading unnecessary files (databases, Python cache, etc.)

```powershell
# Create .gitignore file
@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# LanceDB
my_database/
*.lance
*.arrow

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log

# Environment variables
.env
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

---

## Step 6: Add Files to Git

```powershell
# Add all files to staging area
git add .

# Check what will be committed
git status

# Commit the files
git commit -m "Initial commit: Complete LanceDB learning project with 9 tutorials"
```
## Step 8: Push to GitHub

```powershell
# Rename branch to 'main' (if needed)
git branch -M main

# Push to GitHub (first time)
git push -u origin main
```

### Authentication Options:

**IMPORTANT:** You'll need to authenticate with your PERSONAL GitHub account (not corporate).

#### Option 1: GitHub CLI (Recommended for Managing Multiple Accounts)
```powershell
# Install GitHub CLI if not already installed
winget install --id GitHub.cli

# Authenticate with your PERSONAL GitHub account
gh auth login

# Follow prompts:
# - Choose: GitHub.com (not GitHub Enterprise)
# - Choose: HTTPS
# - Authenticate via web browser
# - This creates a separate token for personal use
```

#### Option 2: Personal Access Token (PAT)
1. Go to your **PERSONAL** GitHub account: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name: "LanceDB Learning Project"
4. Select scopes: 
   - ✅ `repo` (full control of private repositories)
5. Click **"Generate token"**
6. **COPY THE TOKEN** (you won't see it again!)
7. When prompted for password during `git push`, paste the token

#### Option 3: Git Credential Manager
Windows Git Credential Manager will prompt you to sign in via browser:
- Make sure to sign in with your **PERSONAL** GitHub account
- NOT your corporate account

**Storage Location of Credentials:**
- Windows Credential Manager stores Git credentials
- You can have multiple GitHub accounts stored
- Each repository can use different credentials
```

**Example:**
```powershell
git remote add origin https://github.com/johnsmith/lancedb-learning.git
```

---

## Step 8: Push to GitHub

```powershell
# Rename branch to 'main' (if needed)
git branch -M main

# Push to GitHub
## Troubleshooting

### Issue 1: Wrong GitHub Account Used (Corporate Instead of Personal)

**Symptoms:** 
- Push succeeds but commits show corporate name/email
- Repository appears under wrong account

**Solution:**
```powershell
# Check current config
cd C:\Learning\LanceDB
git config user.name
git config user.email

# If showing corporate credentials, reset them:
git config user.name "Your Personal Name"
git config user.email "your.personal@gmail.com"

# Amend last commit with correct author
git commit --amend --reset-author --no-edit

# Force push (if already pushed with wrong credentials)
git push --force
```

### Issue 2: Authentication Failed
**Solution:** Use Personal Access Token (PAT)

1. Go to **YOUR PERSONAL** GitHub: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Generate and copy the token
5. Use token as password when pushing

**Clear cached credentials if needed:**
```powershell
# Remove cached GitHub credentials
git credential-manager-core erase
# Or open Windows Credential Manager and remove GitHub entries
```

1. Go to your GitHub repository URL
2. Refresh the page
3. You should see all your files uploaded!

---

## Alternative: Using GitHub Desktop (GUI Method)

If you prefer a graphical interface:

### Install GitHub Desktop:
1. Download from: https://desktop.github.com
2. Install and sign in to your GitHub account

### Upload Project:
1. Click **File** → **Add local repository**
2. Browse to `C:\Learning\LanceDB`
3. Click **Create repository** (if prompted)
4. Click **Publish repository**
5. Choose repository name and visibility
6. Click **Publish repository**

---

## Troubleshooting

### Issue 1: Authentication Failed
**Solution:** Use Personal Access Token (PAT)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Generate and copy the token
5. Use token as password when pushing

### Issue 2: Remote Already Exists
```powershell
# Remove existing remote
git remote remove origin

# Add correct remote
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
```

### Issue 3: Large Files Error
```powershell
# Remove large database files
git rm --cached -r my_database/

# Add to .gitignore and commit
git add .gitignore
git commit -m "Update .gitignore to exclude database files"
git push
```
## Managing Multiple Git Identities (Office vs Personal)

### Best Practice: Per-Directory Git Configuration

Create a dedicated folder structure for personal projects:

```powershell
# Create personal projects directory (if not exists)
New-Item -ItemType Directory -Path "C:\Personal" -Force
```

**Option A: Configure Git for All Personal Projects**

Create `C:\Personal\.gitconfig`:
```powershell
@"
[user]
    name = Your Personal Name
    email = your.personal@gmail.com
"@ | Out-File -FilePath C:\Personal\.gitconfig -Encoding utf8
```

Then update your global Git config to use conditional includes:
```powershell
# Edit global Git config
notepad $env:USERPROFILE\.gitconfig
```

Add this at the end:
```
[includeIf "gitdir:C:/Personal/"]
    path = C:/Personal/.gitconfig
```

Now all repos under `C:\Personal\` automatically use personal credentials!

**Option B: Set Config Per Repository (What We Did)**
For each personal project:
```powershell
cd C:\Learning\LanceDB
git config user.name "Your Personal Name"
git config user.email "your.personal@gmail.com"
```

### Verify Which Identity is Active

```powershell
# In office repository
cd C:\Work\SomeOfficeProject
git config user.email  # Shows: yourname@company.com

## Quick Reference Commands

```powershell
# Check status
git status

# Verify you're using personal credentials (important!)
git config user.name
git config user.email

# Add all changes
git add .

# Commit changes (uses local config - your personal name/email)
git commit -m "Your message here"

# Push to GitHub (uses personal GitHub authentication)
git push

# Pull latest changes
git pull

# View commit history (shows author info)
git log --oneline

# View detailed commit info
git log --pretty=format:"%h - %an <%ae> : %s" -5

# Create new branch
git checkout -b new-feature

# Switch branches
git checkout main
```

## Configuration Cheat Sheet

```powershell
# View all configurations
git config --list --show-origin

# Global (office) settings - DO NOT CHANGE
git config --global user.name
git config --global user.email

# Local (this project) settings - Set for personal projects
git config --local user.name "Personal Name"
git config --local user.email "personal@email.com"

# Check which config is being used
git config user.name   # Shows local if set, otherwise global
git config user.email  # Shows local if set, otherwise global
``` COMPLETION_SUMMARY.md
├── GITHUB_SETUP.md
├── 01_basic_operations.py
├── 02_vector_search.py
├── 03_filtering_queries.py
├── 04_indexing.py
├── 05_full_text_search.py
├── 06_versioning.py
├── 07_rag_system.py
├── 08_semantic_search.py
└── 09_best_practices.py
```

**Note:** Database files (`my_database/`) will NOT be uploaded (excluded in .gitignore)

---

## Updating Your Repository (Future Changes)

When you make changes to files:

```powershell
# Check what changed
git status

# Add changed files
git add .

# Commit with a message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## Making Repository Look Professional

### Add Topics/Tags:
1. Go to your repository on GitHub
2. Click "About" (⚙️ gear icon)
3. Add topics: `lancedb`, `vector-database`, `python`, `machine-learning`, `rag`, `semantic-search`, `tutorial`

### Update README.md:
Add badges to make it look professional:

```markdown
# LanceDB Learning Project

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LanceDB](https://img.shields.io/badge/LanceDB-Latest-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Complete hands-on tutorial project for learning LanceDB...
```

### Add License:
1. Click "Add file" → "Create new file"
2. Name it `LICENSE`
3. Choose a license template (MIT is popular for tutorials)

---

## Quick Reference Commands

```powershell
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push

# Pull latest changes
git pull

# View commit history
git log --oneline

# Create new branch
git checkout -b new-feature

# Switch branches
git checkout main
```

---

## Next Steps After Upload

1. ✅ Share your repository URL with others
2. ✅ Add a comprehensive README with screenshots
3. ✅ Create a GitHub Pages site for documentation
4. ✅ Enable GitHub Actions for CI/CD
5. ✅ Star repositories you find useful
6. ✅ Add project to your GitHub profile

---

## GitHub Repository URL Format

Your repository will be accessible at:
```
https://github.com/YOUR_USERNAME/lancedb-learning
```

**Example:**
```
https://github.com/johnsmith/lancedb-learning
```

---

## Support

- **Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com
- **GitHub Support:** https://support.github.com

---

**🎉 Congratulations! Your LanceDB learning project is now on GitHub!**
