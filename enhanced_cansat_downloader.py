#!/usr/bin/env python3
"""
Enhanced UAV Dataset Downloader for CanSat Educational Program
Optimized for daylight, high-resolution aerial images from multiple datasets
Automatically cleans up zip files and filters for quality
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image, ImageStat
import requests
import time

# CanSat-specific dataset preferences
CANSAT_DATASETS = [
    {
        'ref': 'bulentsiyah/semantic-drone-dataset',
        'priority': 10,
        'note': 'High-quality semantic drone dataset with daylight images',
        'expected_type': 'daylight_aerial'
    },
    {
        'ref': 'dasmehdixtr/drone-dataset-uav',
        'priority': 9,
        'note': 'UAV collection with varied aerial perspectives',
        'expected_type': 'mixed_aerial'
    },
    {
        'ref': 'tensorflow/aerial-images',
        'priority': 8,
        'note': 'High-resolution aerial imagery collection',
        'expected_type': 'daylight_aerial'
    },
    {
        'ref': 'kmader/drone-images',
        'priority': 7,
        'note': 'Diverse drone photography dataset',
        'expected_type': 'mixed_aerial'
    },
    {
        'ref': 'sshikamaru/uav-aerial-dataset',
        'priority': 6,
        'note': 'UAV surveillance and monitoring imagery',
        'expected_type': 'surveillance_aerial'
    }
]

# Enhanced search terms for CanSat program
CANSAT_SEARCH_TERMS = [
    "aerial photography",
    "drone surveillance", 
    "uav dataset",
    "aerial monitoring",
    "quadcopter images",
    "aerial reconnaissance",
    "satellite simulation",
    "aerial vehicle dataset",
    "remote sensing imagery",
    "aerial survey data"
]

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔧 Setting up Kaggle API...")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    if not kaggle_config.exists():
        print("❌ Kaggle API credentials not found!")
        return False
    
    os.chmod(kaggle_config, 0o600)
    print("✅ Kaggle API credentials configured!")
    return True

def is_daylight_image(image_path: Path) -> bool:
    """Check if image is taken in daylight (not thermal/infrared)"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate image statistics
            stat = ImageStat.Stat(img)
            
            # Check for thermal/infrared characteristics
            # Thermal images often have:
            # 1. Limited color range (mostly grayscale or false color)
            # 2. Unusual color distributions
            # 3. High contrast in specific channels
            
            r_mean, g_mean, b_mean = stat.mean
            r_std, g_std, b_std = stat.stddev
            
            # Daylight images typically have:
            # - Balanced color channels
            # - Good color variation
            # - Natural color distributions
            
            # Check for color balance (daylight images have more balanced RGB)
            color_balance = abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(r_mean - b_mean)
            
            # Check for color variation (thermal images often lack color diversity)
            color_variation = (r_std + g_std + b_std) / 3
            
            # Heuristics for daylight detection
            is_balanced = color_balance < 60  # Colors are reasonably balanced
            has_variation = color_variation > 15  # Has reasonable color variation
            not_extreme = all(20 < mean < 235 for mean in [r_mean, g_mean, b_mean])  # Not extremely dark/bright
            
            return is_balanced and has_variation and not_extreme
            
    except Exception as e:
        print(f"⚠️ Error analyzing {image_path}: {e}")
        return False

def check_image_quality(image_path: Path, min_resolution: Tuple[int, int] = (640, 480)) -> Dict:
    """Check image quality and characteristics for CanSat suitability"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Quality metrics
            quality_info = {
                'width': width,
                'height': height,
                'total_pixels': width * height,
                'aspect_ratio': width / height,
                'is_high_res': width >= min_resolution[0] and height >= min_resolution[1],
                'is_daylight': is_daylight_image(image_path),
                'format': img.format,
                'mode': img.mode,
                'suitable_for_cansat': False
            }
            
            # CanSat suitability criteria
            is_good_resolution = quality_info['total_pixels'] >= 300000  # At least 300K pixels
            is_good_aspect = 1.0 <= quality_info['aspect_ratio'] <= 2.5  # Reasonable aspect ratio
            is_rgb = img.mode in ['RGB', 'RGBA']
            
            quality_info['suitable_for_cansat'] = (
                is_good_resolution and 
                is_good_aspect and 
                is_rgb and 
                quality_info['is_daylight']
            )
            
            return quality_info
            
    except Exception as e:
        return {
            'suitable_for_cansat': False,
            'error': str(e)
        }

def search_enhanced_datasets():
    """Search for datasets with CanSat-specific terms"""
    print("🔍 Searching for CanSat-suitable datasets...")
    
    datasets = []
    
    # Search with CanSat-specific terms
    for term in CANSAT_SEARCH_TERMS:
        try:
            print(f"   Searching for: {term}")
            os.system(f'kaggle datasets list -s "{term}" --csv > temp_search_{term.replace(" ", "_")}.csv')
            
            csv_file = f'temp_search_{term.replace(" ", "_")}.csv'
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
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
                
                os.remove(csv_file)
                
        except Exception as e:
            print(f"   Error searching for {term}: {e}")
    
    return datasets

def score_datasets_for_cansat(datasets: List[Dict]) -> List[Dict]:
    """Score datasets specifically for CanSat educational use"""
    print("📊 Scoring datasets for CanSat suitability...")
    
    scored_datasets = []
    
    for dataset in datasets:
        score = 0
        title_lower = dataset['title'].lower()
        
        # CanSat-specific scoring
        # High priority terms
        if any(term in title_lower for term in ['drone', 'uav', 'aerial']):
            score += 15
        if any(term in title_lower for term in ['surveillance', 'monitoring', 'survey']):
            score += 10
        if any(term in title_lower for term in ['high resolution', 'hd', '4k']):
            score += 8
        if any(term in title_lower for term in ['daylight', 'rgb', 'visible']):
            score += 8
        
        # Educational/research indicators
        if any(term in title_lower for term in ['dataset', 'collection', 'academic', 'research']):
            score += 5
        if any(term in title_lower for term in ['semantic', 'annotated', 'labeled']):
            score += 5
        
        # Negative scoring for unwanted types
        if any(term in title_lower for term in ['thermal', 'infrared', 'ir', 'night', 'dark']):
            score -= 10
        if any(term in title_lower for term in ['medical', 'x-ray', 'indoor']):
            score -= 5
        
        # Size and popularity bonus
        download_count = dataset.get('downloadCount', 0)
        if isinstance(download_count, str):
            try:
                download_count = int(download_count)
            except:
                download_count = 0
        
        score += min(download_count // 50, 10)  # Max 10 points for popularity
        
        dataset['cansat_score'] = score
        if score > 8:  # Only include reasonably suitable datasets
            scored_datasets.append(dataset)
    
    # Sort by CanSat score
    scored_datasets.sort(key=lambda x: x['cansat_score'], reverse=True)
    
    return scored_datasets

def download_multiple_datasets(datasets: List[Dict], max_datasets: int = 5, download_dir: str = "downloads") -> List[str]:
    """Download multiple datasets for diversity"""
    print(f"📥 Downloading up to {max_datasets} datasets...")
    
    Path(download_dir).mkdir(exist_ok=True)
    successful_downloads = []
    
    # Add priority datasets first
    priority_refs = [d['ref'] for d in CANSAT_DATASETS]
    
    # Combine priority and searched datasets
    all_datasets = []
    
    # Add priority datasets first
    for priority_dataset in CANSAT_DATASETS:
        all_datasets.append({
            'ref': priority_dataset['ref'],
            'title': f"Priority: {priority_dataset['note']}",
            'cansat_score': priority_dataset['priority'] + 10  # Boost priority
        })
    
    # Add searched datasets
    all_datasets.extend(datasets)
    
    # Remove duplicates and sort
    seen_refs = set()
    unique_datasets = []
    for dataset in all_datasets:
        if dataset['ref'] not in seen_refs and dataset['ref']:
            seen_refs.add(dataset['ref'])
            unique_datasets.append(dataset)
    
    unique_datasets.sort(key=lambda x: x.get('cansat_score', 0), reverse=True)
    
    for i, dataset in enumerate(unique_datasets[:max_datasets]):
        try:
            print(f"\n📦 Downloading {i+1}/{max_datasets}: {dataset['ref']}")
            print(f"   Title: {dataset.get('title', 'N/A')}")
            print(f"   Score: {dataset.get('cansat_score', 'N/A')}")
            
            result = os.system(f'kaggle datasets download -d {dataset["ref"]} -p {download_dir}')
            
            if result == 0:
                print(f"✅ Successfully downloaded {dataset['ref']}")
                successful_downloads.append(dataset['ref'])
            else:
                print(f"❌ Failed to download {dataset['ref']}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset['ref']}: {e}")
    
    return successful_downloads

def extract_and_filter_images(download_dir: str = "downloads", target_dir: str = "cansat_images", target_count: int = 1000) -> int:
    """Extract and filter images with CanSat-specific quality checks"""
    print("📦 Extracting and filtering images for CanSat use...")
    
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    download_path = Path(download_dir)
    image_count = 0
    processed_count = 0
    skipped_count = 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Process all zip files
    zip_files = list(download_path.glob("*.zip"))
    print(f"📂 Found {len(zip_files)} zip files to process")
    
    for zip_file in zip_files:
        try:
            print(f"\n📂 Processing {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                temp_dir = download_path / f"temp_{zip_file.stem}"
                zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_files = []
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        image_files.append(file_path)
                
                print(f"   Found {len(image_files)} image files")
                
                # Process images with quality filtering
                for file_path in image_files:
                    processed_count += 1
                    
                    # Check image quality
                    quality_info = check_image_quality(file_path)
                    
                    if quality_info.get('suitable_for_cansat', False):
                        try:
                            # Copy to target directory
                            new_name = f"cansat_image_{image_count:04d}{file_path.suffix.lower()}"
                            new_path = target_path / new_name
                            shutil.copy2(file_path, new_path)
                            image_count += 1
                            
                            if image_count % 50 == 0:
                                print(f"   ✅ Copied {image_count} suitable images")
                            
                            # Stop if we reach target
                            if image_count >= target_count:
                                print(f"🎯 Reached target of {target_count} images!")
                                break
                                
                        except Exception as e:
                            print(f"   ⚠️ Error copying {file_path}: {e}")
                    else:
                        skipped_count += 1
                        if 'error' not in quality_info:
                            # Show reason for skipping
                            reasons = []
                            if not quality_info.get('is_high_res', False):
                                reasons.append("low resolution")
                            if not quality_info.get('is_daylight', False):
                                reasons.append("not daylight")
                            if quality_info.get('aspect_ratio', 0) > 2.5 or quality_info.get('aspect_ratio', 0) < 1.0:
                                reasons.append("poor aspect ratio")
                            
                            if processed_count % 100 == 0:
                                print(f"   ⚠️ Skipped image: {', '.join(reasons) if reasons else 'quality check failed'}")
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                # Stop if we have enough images
                if image_count >= target_count:
                    break
        
        except Exception as e:
            print(f"❌ Error processing {zip_file}: {e}")
    
    # Clean up zip files
    print("\n🧹 Cleaning up zip files...")
    for zip_file in zip_files:
        try:
            zip_file.unlink()
            print(f"   🗑️ Deleted {zip_file.name}")
        except Exception as e:
            print(f"   ⚠️ Could not delete {zip_file.name}: {e}")
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total images processed: {processed_count}")
    print(f"   Suitable images kept: {image_count}")
    print(f"   Images skipped: {skipped_count}")
    print(f"   Success rate: {(image_count/processed_count*100):.1f}%" if processed_count > 0 else "   No images processed")
    
    return image_count

def create_cansat_dataset_info(image_count: int, target_dir: str = "cansat_images"):
    """Create comprehensive dataset information for CanSat program"""
    info = {
        "dataset_name": "CanSat UAV Training Dataset",
        "total_images": image_count,
        "purpose": "Educational CanSat program - aerial imagery simulation",
        "target_altitude": "100-200 meters (simulated satellite view)",
        "image_characteristics": {
            "type": "Daylight aerial photography",
            "min_resolution": "640x480 pixels",
            "format": "Various (JPG, PNG)",
            "color_space": "RGB",
            "quality_filtered": True
        },
        "cansat_suitability": {
            "earth_observation": "High",
            "terrain_recognition": "High", 
            "altitude_simulation": "Medium-High",
            "educational_value": "High"
        },
        "created_date": "2025-08-10",
        "source": "Multiple Kaggle datasets",
        "filtering_applied": [
            "Daylight images only (no thermal/infrared)",
            "High resolution filtering",
            "Aspect ratio validation",
            "Color space verification",
            "Quality assessment"
        ]
    }
    
    info_path = Path(target_dir) / "cansat_dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📋 CanSat dataset info saved to {info_path}")

def main():
    """Enhanced main function for CanSat-optimized downloads"""
    print("🛰️  Enhanced CanSat UAV Dataset Downloader")
    print("=" * 50)
    print("Optimized for educational CanSat programs")
    print("Features: Daylight images, high-resolution, multiple datasets, auto-cleanup")
    print()
    
    # Step 1: Setup
    if not setup_kaggle_api():
        print("❌ Please setup Kaggle API credentials first!")
        return
    
    # Step 2: Search for enhanced datasets
    print("🔍 Searching for CanSat-suitable datasets...")
    datasets = search_enhanced_datasets()
    
    if not datasets:
        print("⚠️ No datasets found from search, using priority datasets only")
        datasets = []
    
    print(f"📊 Found {len(datasets)} datasets from search")
    
    # Step 3: Score datasets for CanSat use
    scored_datasets = score_datasets_for_cansat(datasets)
    
    print(f"\n🎯 Top CanSat-suitable datasets:")
    print("-" * 40)
    for i, dataset in enumerate(scored_datasets[:5], 1):
        print(f"{i}. {dataset['title']}")
        print(f"   📦 {dataset['ref']}")
        print(f"   🎯 CanSat Score: {dataset['cansat_score']}")
        print()
    
    # Step 4: Download multiple datasets
    print("📥 Starting multi-dataset download...")
    successful_downloads = download_multiple_datasets(scored_datasets, max_datasets=5)
    
    if not successful_downloads:
        print("❌ No datasets were successfully downloaded!")
        return
    
    print(f"✅ Successfully downloaded {len(successful_downloads)} datasets")
    
    # Step 5: Extract and filter with quality checks
    print("\n🔍 Extracting and filtering for CanSat quality...")
    image_count = extract_and_filter_images(target_dir="cansat_images", target_count=1000)
    
    # Step 6: Create CanSat-specific dataset info
    create_cansat_dataset_info(image_count)
    
    print("\n🎉 CanSat dataset preparation complete!")
    print(f"📊 High-quality images collected: {image_count}")
    print(f"📁 Images saved in: cansat_images/")
    print(f"🛰️  Ready for CanSat educational program!")
    
    if image_count < 1000:
        print(f"\n💡 Tips to get more images:")
        print("   - Run the script again to try additional datasets")
        print("   - Check if some datasets require competition participation")
        print("   - Consider adjusting quality filters if needed")

if __name__ == "__main__":
    main()
