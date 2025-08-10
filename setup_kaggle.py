#!/usr/bin/env python3
"""
Kaggle API Setup Helper
Helps users set up Kaggle API credentials for downloading datasets
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials"""
    print("🔧 Kaggle API Credentials Setup")
    print("=" * 40)
    
    print("\n📋 To use the Kaggle API, you need to:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print()
    
    # Check if credentials already exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    if kaggle_config.exists():
        print("✅ Kaggle credentials already found!")
        print(f"📁 Location: {kaggle_config}")
        
        # Verify permissions
        stat = kaggle_config.stat()
        if oct(stat.st_mode)[-3:] != '600':
            print("🔒 Setting correct permissions...")
            os.chmod(kaggle_config, 0o600)
            print("✅ Permissions updated!")
        else:
            print("✅ Permissions are correct!")
        
        return True
    
    print("❌ Kaggle credentials not found!")
    print("\nChoose an option:")
    print("1. I have downloaded kaggle.json and want to set it up")
    print("2. I need to download kaggle.json first")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        return setup_existing_json()
    elif choice == "2":
        return show_download_instructions()
    else:
        print("👋 Exiting setup...")
        return False

def setup_existing_json():
    """Setup when user has already downloaded kaggle.json"""
    print("\n📂 Setting up existing kaggle.json file...")
    
    # Common locations where users might have saved the file
    common_locations = [
        Path.home() / "Downloads" / "kaggle.json",
        Path.cwd() / "kaggle.json",
        Path.home() / "kaggle.json"
    ]
    
    kaggle_json_path = None
    
    # Check common locations
    for location in common_locations:
        if location.exists():
            print(f"📁 Found kaggle.json at: {location}")
            kaggle_json_path = location
            break
    
    if not kaggle_json_path:
        # Ask user for custom path
        custom_path = input("Enter the full path to your kaggle.json file: ").strip()
        kaggle_json_path = Path(custom_path)
        
        if not kaggle_json_path.exists():
            print(f"❌ File not found: {kaggle_json_path}")
            return False
    
    # Validate the JSON file
    try:
        with open(kaggle_json_path, 'r') as f:
            creds = json.load(f)
        
        if 'username' not in creds or 'key' not in creds:
            print("❌ Invalid kaggle.json format! Must contain 'username' and 'key'")
            return False
        
        print("✅ Valid kaggle.json file found!")
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON format in kaggle.json!")
        return False
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy the file
    target_path = kaggle_dir / 'kaggle.json'
    import shutil
    shutil.copy2(kaggle_json_path, target_path)
    
    # Set correct permissions
    os.chmod(target_path, 0o600)
    
    print(f"✅ Kaggle credentials installed at: {target_path}")
    print("🔒 Permissions set to 600 (read/write for owner only)")
    print("🎉 Setup complete! You can now use the Kaggle API.")
    
    return True

def show_download_instructions():
    """Show detailed download instructions"""
    print("\n📋 Step-by-step instructions to get your Kaggle API token:")
    print("-" * 50)
    print("1. 🌐 Open your web browser and go to: https://www.kaggle.com")
    print("2. 🔑 Log in to your Kaggle account (create one if you don't have it)")
    print("3. 👤 Click on your profile picture in the top right corner")
    print("4. ⚙️  Select 'Account' from the dropdown menu")
    print("5. 📜 Scroll down to the 'API' section")
    print("6. 🔐 Click 'Create New API Token'")
    print("7. 💾 This will download a file called 'kaggle.json'")
    print("8. 📁 Remember where you saved this file!")
    print("9. 🔄 Run this setup script again and choose option 1")
    print()
    print("💡 Pro tip: The kaggle.json file contains your API credentials.")
    print("   Keep it secure and don't share it with others!")
    
    return False

def test_kaggle_connection():
    """Test if Kaggle API is working"""
    print("\n🧪 Testing Kaggle API connection...")
    
    try:
        result = os.system("kaggle datasets list --page-size 1 > /dev/null 2>&1")
        if result == 0:
            print("✅ Kaggle API is working correctly!")
            return True
        else:
            print("❌ Kaggle API test failed!")
            return False
    except Exception as e:
        print(f"❌ Error testing Kaggle API: {e}")
        return False

def main():
    """Main setup function"""
    print("🚁 UAV Dataset Downloader - Kaggle Setup")
    print("=" * 45)
    
    if setup_kaggle_credentials():
        test_kaggle_connection()
        print("\n🎯 You're ready to download UAV datasets!")
        print("Run: python kaggle_uav_downloader.py")
    else:
        print("\n❌ Setup incomplete. Please try again.")

if __name__ == "__main__":
    main()
