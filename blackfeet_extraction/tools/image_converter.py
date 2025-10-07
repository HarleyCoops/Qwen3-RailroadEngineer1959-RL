"""
Image Format Converter for Blackfeet Dictionary Pages

Handles JP2 (JPEG 2000) files and converts them to formats compatible with
the vision models. JP2 is commonly used for archival scans due to superior
compression and quality preservation.
"""

from pathlib import Path
from typing import List, Optional
from PIL import Image
import base64
from io import BytesIO


class ImageConverter:
    """Convert and prepare dictionary images for processing."""

    def __init__(
        self,
        input_dir: str = "data/dictionary_pages",
        output_dir: str = "data/processed_images",
        output_format: str = "JPEG",
        quality: int = 95,
    ):
        """
        Initialize image converter.

        Args:
            input_dir: Directory containing source images (JP2, etc.)
            output_dir: Directory for converted images
            output_format: Target format (JPEG, PNG)
            quality: Output quality (1-100 for JPEG)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format.upper()
        self.quality = quality

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_jp2_to_jpeg(
        self,
        jp2_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Convert a single JP2 file to JPEG.

        Args:
            jp2_path: Path to JP2 file
            output_path: Optional output path (auto-generated if None)

        Returns:
            Path to converted image
        """
        if not jp2_path.exists():
            raise FileNotFoundError(f"JP2 file not found: {jp2_path}")

        # Auto-generate output path if not provided
        if output_path is None:
            output_name = jp2_path.stem + ".jpg"
            output_path = self.output_dir / output_name

        print(f"Converting: {jp2_path.name} -> {output_path.name}")

        try:
            # Open JP2 file with Pillow
            with Image.open(jp2_path) as img:
                # Convert to RGB if needed (JP2 can be in various color modes)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as JPEG
                img.save(output_path, "JPEG", quality=self.quality, optimize=True)

            print("  OK Converted successfully")
            return output_path

        except Exception as e:
            print(f"  ERROR converting {jp2_path.name}: {e}")
            raise

    def convert_all_jp2_files(self) -> List[Path]:
        """
        Convert all JP2 files in input directory.

        Returns:
            List of paths to converted images
        """
        jp2_files = sorted(self.input_dir.glob("*.jp2"))

        if not jp2_files:
            print(f"WARNING: No JP2 files found in {self.input_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"Converting {len(jp2_files)} JP2 files to JPEG")
        print(f"{'='*60}\n")

        converted_paths = []
        for jp2_file in jp2_files:
            try:
                converted_path = self.convert_jp2_to_jpeg(jp2_file)
                converted_paths.append(converted_path)
            except Exception as e:
                print(f"WARNING: Skipping {jp2_file.name} due to error: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Conversion complete: {len(converted_paths)}/{len(jp2_files)} successful")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        return converted_paths

    def get_image_info(self, image_path: Path) -> dict:
        """
        Get information about an image file.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image metadata
        """
        with Image.open(image_path) as img:
            return {
                "path": str(image_path),
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size_mb": image_path.stat().st_size / (1024 * 1024),
            }

    def optimize_for_api(
        self,
        image_path: Path,
        max_size: int = 2048,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Optimize image for API submission (reduce size if needed).

        Many vision APIs have size limits. This resizes images while
        maintaining aspect ratio and quality.

        Args:
            image_path: Path to image
            max_size: Maximum dimension (width or height)
            output_path: Optional output path

        Returns:
            Path to optimized image
        """
        if output_path is None:
            output_path = self.output_dir / f"optimized_{image_path.name}"

        with Image.open(image_path) as img:
            # Check if resizing is needed
            if max(img.size) > max_size:
                print(f"Resizing {image_path.name} from {img.size} to fit {max_size}px")

                # Calculate new size maintaining aspect ratio
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)

                # Resize with high-quality resampling
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save optimized
            img.save(output_path, "JPEG", quality=self.quality, optimize=True)

        return output_path

    def jp2_to_base64(self, jp2_path: Path) -> tuple[str, str]:
        """
        Convert JP2 directly to base64 (for API submission).

        Args:
            jp2_path: Path to JP2 file

        Returns:
            Tuple of (base64_string, media_type)
        """
        # First convert to JPEG in memory
        with Image.open(jp2_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to BytesIO buffer as JPEG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)

            # Encode to base64
            base64_string = base64.b64encode(buffer.read()).decode("utf-8")

        return base64_string, "image/jpeg"


def main():
    """Example usage."""
    import sys

    # Initialize converter
    converter = ImageConverter(
        input_dir="data/dictionary_pages",  # Where your JP2 files are
        output_dir="data/processed_images",  # Where JPEGs will go
        quality=95,  # High quality for OCR
    )

    # Check for JP2 files
    jp2_files = list(Path("data/dictionary_pages").glob("*.jp2"))

    if not jp2_files:
        print("No JP2 files found in data/dictionary_pages/")
        print("\nPlease place your JP2 dictionary pages in:")
        print("  data/dictionary_pages/")
        print("\nExample:")
        print("  data/dictionary_pages/page_001.jp2")
        print("  data/dictionary_pages/page_002.jp2")
        print("  etc.")
        sys.exit(1)

    print(f"Found {len(jp2_files)} JP2 files")

    # Option 1: Convert all JP2 files
    converted = converter.convert_all_jp2_files()

    # Option 2: Show info about first converted file
    if converted:
        info = converter.get_image_info(converted[0])
        print("\nFirst converted image info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # Option 3: Optimize for API (if needed)
    # Some images from archives can be very large
    if converted:
        optimized = converter.optimize_for_api(
            converted[0],
            max_size=2048,  # Reasonable size for OCR
        )
        print(f"\nOptimized image saved to: {optimized}")


if __name__ == "__main__":
    main()
