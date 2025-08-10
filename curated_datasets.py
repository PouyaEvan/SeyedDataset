#!/usr/bin/env python3
"""
Curated UAV Dataset References
Pre-selected high-quality UAV datasets that are likely to contain images from 100-200m height
"""

# Curated list of promising UAV datasets on Kaggle
CURATED_UAV_DATASETS = [
    {
        'ref': 'dasmehdixtr/drone-dataset-uav',
        'title': 'Drone Dataset UAV',
        'description': 'Collection of drone/UAV images for computer vision',
        'expected_images': '500+',
        'note': 'Good variety of aerial shots'
    },
    {
        'ref': 'bulentsiyah/semantic-drone-dataset',
        'title': 'Semantic Drone Dataset',
        'description': 'Semantic segmentation dataset with drone imagery',
        'expected_images': '400+',
        'note': 'High quality annotated aerial images'
    },
    {
        'ref': 'kmader/drone-images',
        'title': 'Drone Images Collection',
        'description': 'Various drone captured images',
        'expected_images': '300+',
        'note': 'Mixed altitude drone photography'
    },
    {
        'ref': 'sshikamaru/uav-aerial-dataset',
        'title': 'UAV Aerial Dataset',
        'description': 'Aerial images captured by UAVs',
        'expected_images': '600+',
        'note': 'Focus on aerial surveillance data'
    },
    {
        'ref': 'tensorflow/aerial-images',
        'title': 'Aerial Images Dataset',
        'description': 'High-resolution aerial imagery',
        'expected_images': '1000+',
        'note': 'Professional quality aerial shots'
    }
]

# Alternative search terms for finding more datasets
EXTENDED_SEARCH_TERMS = [
    "aerial surveillance",
    "remote sensing",
    "satellite imagery",
    "aerial photography",
    "quadcopter images",
    "multirotor dataset",
    "aerial vehicle",
    "unmanned aircraft",
    "aerial monitoring",
    "drone surveillance",
    "cansat dataset",
    "high altitude balloon",
    "aerial reconnaissance"
]

def print_curated_datasets():
    """Print the curated dataset list"""
    print("🎯 Curated UAV Datasets for CanSat Program")
    print("=" * 50)
    
    for i, dataset in enumerate(CURATED_UAV_DATASETS, 1):
        print(f"\n{i}. {dataset['title']}")
        print(f"   📦 Reference: {dataset['ref']}")
        print(f"   📝 Description: {dataset['description']}")
        print(f"   📊 Expected Images: {dataset['expected_images']}")
        print(f"   💡 Note: {dataset['note']}")

def print_search_terms():
    """Print extended search terms"""
    print("\n🔍 Extended Search Terms for UAV Datasets")
    print("-" * 40)
    
    for i, term in enumerate(EXTENDED_SEARCH_TERMS, 1):
        print(f"{i:2d}. {term}")

if __name__ == "__main__":
    print_curated_datasets()
    print_search_terms()
    
    print("\n💡 Usage Tips:")
    print("- Try downloading curated datasets first")
    print("- Use extended search terms if you need more images")
    print("- Some datasets may require Kaggle competition participation")
    print("- Quality varies - the main script will filter and organize images")
