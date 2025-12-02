#!/usr/bin/env python3
"""
Helper script to download and prepare datasets for multi-exercise training.

Usage:
    python download_datasets.py --ucf101          # Download UCF-101
    python download_datasets.py --coco             # Download COCO annotations
    python download_datasets.py --list             # List available datasets
    python download_datasets.py --all              # Download all available datasets
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import train_model functions
sys.path.insert(0, str(Path(__file__).parent))

try:
    from train_model import (
        download_ucf101_all_workouts,
        download_coco_pose_dataset,
        DATA_ROOT
    )
except ImportError as e:
    print(f"Error importing train_model: {e}")
    print("Make sure train_model.py is in the same directory")
    sys.exit(1)


def list_datasets():
    """List all available datasets and their status."""
    print("="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    
    datasets = [
        {
            "name": "UCF-101",
            "description": "Video dataset with multiple exercise types",
            "size": "~50GB",
            "format": "Videos",
            "command": "--ucf101",
            "status": "Available"
        },
        {
            "name": "COCO Pose",
            "description": "Diverse human pose annotations (images)",
            "size": "~20GB",
            "format": "Images + JSON",
            "command": "--coco",
            "status": "Available (annotations auto-download, images manual)"
        },
        {
            "name": "M3GYM",
            "description": "Real gym environment dataset",
            "size": "Large",
            "format": "Videos",
            "command": "Manual (requires approval)",
            "status": "Requires access approval"
        },
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Format: {dataset['format']}")
        print(f"   Download: python download_datasets.py {dataset['command']}")
        print(f"   Status: {dataset['status']}")
    
    print("\n" + "="*60)
    print("For detailed information, see DATASETS_GUIDE.md")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for multi-exercise training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download UCF-101 dataset
  python download_datasets.py --ucf101
  
  # Download COCO annotations
  python download_datasets.py --coco
  
  # List all available datasets
  python download_datasets.py --list
  
  # Download all available datasets
  python download_datasets.py --all
        """
    )
    
    parser.add_argument("--ucf101", action="store_true",
                       help="Download UCF-101 dataset")
    parser.add_argument("--coco", action="store_true",
                       help="Download COCO Pose dataset annotations")
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets")
    parser.add_argument("--all", action="store_true",
                       help="Download all available datasets")
    
    args = parser.parse_args()
    
    if args.list or (not args.ucf101 and not args.coco and not args.all):
        list_datasets()
        return 0
    
    success_count = 0
    total_count = 0
    
    if args.all or args.ucf101:
        total_count += 1
        print("\n" + "="*60)
        print("DOWNLOADING UCF-101 DATASET")
        print("="*60)
        if download_ucf101_all_workouts():
            print("✓ UCF-101 download completed successfully")
            success_count += 1
        else:
            print("✗ UCF-101 download failed")
    
    if args.all or args.coco:
        total_count += 1
        print("\n" + "="*60)
        print("DOWNLOADING COCO POSE DATASET")
        print("="*60)
        if download_coco_pose_dataset():
            print("✓ COCO annotations downloaded successfully")
            print("  Note: Images must be downloaded manually")
            success_count += 1
        else:
            print("✗ COCO download failed")
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("\nSome downloads failed. Check the error messages above.")
        print("For help, see DATASETS_GUIDE.md")
        return 1
    
    if success_count > 0:
        print("\nNext steps:")
        print("1. Review downloaded data in:", DATA_ROOT)
        print("2. For COCO: Download images manually from https://cocodataset.org/#download")
        print("3. Run: python train_model.py --setup-dirs")
        print("4. Fill in scores in data/exercise_scores.csv")
        print("5. Start training: python train_model.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

