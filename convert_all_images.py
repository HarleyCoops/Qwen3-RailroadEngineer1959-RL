#!/usr/bin/env python3
"""
Convert all JP2 files to JPEG for the entire Dakota dictionary
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dakota_extraction.tools.image_converter import ImageConverter


def main():
    print("\n" + "="*70)
    print(" CONVERTING ALL DICTIONARY IMAGES")
    print("="*70)

    jp2_dir = Path("dictionary/grammardictionar00riggrich_jp2")
    output_dir = Path("data/processed_images")

    # Count JP2 files
    jp2_files = sorted(jp2_dir.glob("*.jp2"))
    total_files = len(jp2_files)

    print(f"\nFound {total_files} JP2 files in {jp2_dir}")
    print(f"Output directory: {output_dir}")
    print("\nThis will convert all images to JPEG format.")
    print("Already converted images will be skipped.\n")

    # Create converter
    converter = ImageConverter(
        input_dir=str(jp2_dir),
        output_dir=str(output_dir),
        quality=95
    )

    # Convert all
    print("Converting...")
    converted = converter.convert_all_jp2_files()

    print("\n" + "="*70)
    print(" CONVERSION COMPLETE")
    print("="*70)
    print(f"\nTotal files: {total_files}")
    print(f"Converted: {converted}")
    print(f"Output: {output_dir}/")

    # List first and last files
    jpg_files = sorted(output_dir.glob("*.jpg"))
    if jpg_files:
        print(f"\nFirst image: {jpg_files[0].name}")
        print(f"Last image: {jpg_files[-1].name}")
        print(f"Total JPG files: {len(jpg_files)}")

    print("\nNext step:")
    print("  1. Open data/processed_images/ in your file browser")
    print("  2. Identify which image numbers contain grammar content")
    print("  3. Run: python extract_grammar_pages.py --pages <start>-<end>")
    print()


if __name__ == "__main__":
    main()
