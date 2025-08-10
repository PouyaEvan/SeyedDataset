#!/usr/bin/env python3
"""
UAV Drone Dataset Downloader for CanSat Program
Downloads drone images from Kaggle datasets, specifically looking for images taken from 100-200m height
Target: ~1000 images for CanSat program
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict
import pandas as pd
from PIL import Image
import requests

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔧 Setting up Kaggle API...")
    
    # Check if kaggle.json exists
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    if not kaggle_config.exists():
        print("❌ Kaggle API credentials not found!")
        print("📋 Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/ directory")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Set proper permissions
    os.chmod(kaggle_config, 0o600)
    print("✅ Kaggle API credentials found and configured!")
    return True

def search_uav_datasets():
    """Search for UAV/drone datasets on Kaggle"""
    print("🔍 Searching for UAV/drone datasets...")
    
    # Search terms related to UAV/drones
    search_terms = [
        "uav", "drone", "aerial", "unmanned aerial vehicle",
        "quadcopter", "multirotor", "aerial photography",
        "aerial surveillance", "aerial images"
    ]
    
    datasets = []
    
    for term in search_terms:
        try:
            # Using kaggle API to search datasets
            os.system(f'kaggle datasets list -s "{term}" --csv > temp_search_{term}.csv')
            
            if os.path.exists(f'temp_search_{term}.csv'):
                df = pd.read_csv(f'temp_search_{term}.csv')
                for _, row in df.iterrows():
                    dataset_info = {
                        'ref': row.get('ref', ''),
                        'title': row.get('title', ''),
                        'size': row.get('size', ''),
                        'lastUpdated': row.get('lastUpdated', ''),
                        'downloadCount': row.get('downloadCount', 0),
                        'search_term': term
                    }
                    datasets.append(dataset_info)
                
                os.remove(f'temp_search_{term}.csv')
        except Exception as e:
            print(f"Error searching for {term}: {e}")
    
    return datasets

def recommend_datasets(datasets: List[Dict]) -> List[Dict]:
    """Recommend best datasets for UAV images"""
    print("📊 Analyzing datasets for UAV relevance...")
    
    # Score datasets based on relevance
    scored_datasets = []
    
    for dataset in datasets:
        score = 0
        title_lower = dataset['title'].lower()
        
        # Scoring criteria
        if 'drone' in title_lower or 'uav' in title_lower:
            score += 10
        if 'aerial' in title_lower:
            score += 8
        if 'height' in title_lower or 'altitude' in title_lower:
            score += 5
        if 'surveillance' in title_lower or 'monitoring' in title_lower:
            score += 3
        if 'image' in title_lower or 'photo' in title_lower:
            score += 3
        
        # Prefer datasets with more downloads (popularity indicator)
        download_count = dataset.get('downloadCount', 0)
        if isinstance(download_count, str):
            try:
                download_count = int(download_count)
            except:
                download_count = 0
        
        score += min(download_count // 100, 5)  # Max 5 points for downloads
        
        dataset['relevance_score'] = score
        if score > 5:  # Only include reasonably relevant datasets
            scored_datasets.append(dataset)
    
    # Sort by relevance score
    scored_datasets.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_datasets[:10]  # Return top 10

def download_dataset(dataset_ref: str, download_dir: str = "downloads") -> bool:
    """Download a specific dataset from Kaggle"""
    try:
        print(f"📥 Downloading dataset: {dataset_ref}")
        
        # Create download directory
        Path(download_dir).mkdir(exist_ok=True)
        
        # Download using kaggle API
        result = os.system(f'kaggle datasets download -d {dataset_ref} -p {download_dir}')
        
        if result == 0:
            print(f"✅ Successfully downloaded {dataset_ref}")
            return True
        else:
            print(f"❌ Failed to download {dataset_ref}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {dataset_ref}: {e}")
        return False

def extract_and_organize_images(download_dir: str = "downloads", target_dir: str = "uav_images") -> int:
    """Extract and organize downloaded images"""
    print("📦 Extracting and organizing images...")
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    download_path = Path(download_dir)
    image_count = 0
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Process all zip files in download directory
    for zip_file in download_path.glob("*.zip"):
        try:
            print(f"📂 Extracting {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract to temporary directory
                temp_dir = download_path / f"temp_{zip_file.stem}"
                zip_ref.extractall(temp_dir)
                
                # Find and copy image files
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        try:
                            # Verify it's a valid image
                            with Image.open(file_path) as img:
                                # Copy to target directory with unique name
                                new_name = f"uav_image_{image_count:04d}{file_path.suffix}"
                                new_path = target_path / new_name
                                shutil.copy2(file_path, new_path)
                                image_count += 1
                                
                                print(f"✅ Copied image {image_count}: {new_name}")
                                
                                # Stop if we reach 1000 images
                                if image_count >= 1000:
                                    print("🎯 Reached target of 1000 images!")
                                    break
                                    
                        except Exception as e:
                            print(f"⚠️ Skipping invalid image {file_path}: {e}")
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                if image_count >= 1000:
                    break
                    
        except Exception as e:
            print(f"❌ Error extracting {zip_file}: {e}")
    
    return image_count

def create_dataset_info(image_count: int, target_dir: str = "uav_images"):
    """Create dataset information file"""
    info = {
        "dataset_name": "UAV Drone Images for CanSat Program",
        "total_images": image_count,
        "target_height_range": "100-200 meters",
        "purpose": "CanSat program training data",
        "created_date": "2025-08-10",
        "format": "Various (JPG, PNG, etc.)",
        "source": "Kaggle datasets"
    }
    
    info_path = Path(target_dir) / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📋 Dataset info saved to {info_path}")

def main():
    """Main function to orchestrate the download process"""
    print("🚁 UAV Drone Dataset Downloader for CanSat Program")
    print("=" * 50)
    
    # Step 1: Setup Kaggle API
    if not setup_kaggle_api():
        print("❌ Please setup Kaggle API credentials first!")
        return
    
    # Step 2: Search for UAV datasets
    datasets = search_uav_datasets()
    if not datasets:
        print("❌ No datasets found! Check your internet connection and Kaggle API setup.")
        return
    
    print(f"📊 Found {len(datasets)} total datasets")
    
    # Step 3: Recommend best datasets
    recommended = recommend_datasets(datasets)
    
    print("\n🎯 Top recommended UAV datasets:")
    print("-" * 40)
    for i, dataset in enumerate(recommended[:5], 1):
        print(f"{i}. {dataset['title']}")
        print(f"   Ref: {dataset['ref']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Relevance Score: {dataset['relevance_score']}")
        print()
    
    # Step 4: Download top datasets
    print("📥 Starting downloads...")
    successful_downloads = 0
    
    for dataset in recommended[:3]:  # Download top 3 datasets
        if download_dataset(dataset['ref']):
            successful_downloads += 1
        
        # Check if we might have enough images already
        if successful_downloads >= 2:
            break
    
    if successful_downloads == 0:
        print("❌ No datasets were successfully downloaded!")
        return
    
    # Step 5: Extract and organize images
    image_count = extract_and_organize_images()
    
    # Step 6: Create dataset info
    create_dataset_info(image_count)
    
    print("\n🎉 Download and organization complete!")
    print(f"📊 Total images collected: {image_count}")
    print(f"📁 Images saved in: uav_images/")
    
    if image_count < 1000:
        print(f"⚠️ Only {image_count} images found. You may want to:")
        print("   - Try more datasets")
        print("   - Search for additional keywords")
        print("   - Consider supplementing with other sources")

if __name__ == "__main__":
    main()
