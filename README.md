# SeyedDataset - UAV Drone Images for CanSat Program

This project automatically downloads UAV/drone images from Kaggle datasets, specifically targeting images taken from 100-200 meters height for CanSat program training data.

## 🎯 Goal
Download approximately 1000 UAV drone pictures from Kaggle datasets for CanSat program use.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API Credentials
```bash
python setup_kaggle.py
```
Follow the interactive setup to configure your Kaggle API credentials.

### 3. Download UAV Images
```bash
python kaggle_uav_downloader.py
```

## 📋 What You Need

### Kaggle Account & API Token
1. Create a free account at [Kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download the `kaggle.json` file
4. Run the setup script to configure it

## 📁 Output Structure
```
uav_images/
├── uav_image_0001.jpg
├── uav_image_0002.png
├── ...
├── uav_image_1000.jpg
└── dataset_info.json
```

## 🔍 Search Criteria
The script searches for datasets containing:
- UAV/drone imagery
- Aerial photography
- Images preferably from 100-200m height
- High-quality training data suitable for CanSat programs

## 📊 Features
- ✅ Automatic dataset discovery and ranking
- ✅ Smart filtering for UAV-relevant content
- ✅ Image validation and format conversion
- ✅ Progress tracking and error handling
- ✅ Organized output with metadata
- ✅ Target of ~1000 images

## 🛠️ Files Description
- `kaggle_uav_downloader.py` - Main downloader script
- `setup_kaggle.py` - Interactive Kaggle API setup
- `requirements.txt` - Python dependencies
- `uav_images/` - Downloaded and organized images (created after run)

## ⚠️ Notes
- Requires active internet connection
- Some datasets may require Kaggle competition participation
- Script will automatically handle image format conversion
- Downloads are organized and renamed for consistency

## 🚁 CanSat Program
This dataset is specifically curated for CanSat (Can Satellite) programs, providing aerial imagery that simulates the view from small satellites or high-altitude balloons.