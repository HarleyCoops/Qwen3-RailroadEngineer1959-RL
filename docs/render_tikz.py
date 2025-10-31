#!/usr/bin/env python3
"""
Render TikZ LaTeX diagram to PNG image.
Requires: pdflatex (or lualatex) and pdf2image (pip install pdf2image)
"""

import subprocess
import sys
import os
from pathlib import Path

def render_tikz_to_png(tex_file, output_png, dpi=300):
    """
    Compile TikZ LaTeX file and convert to PNG.
    
    Args:
        tex_file: Path to .tex file
        output_png: Path for output PNG file
        dpi: Resolution for PNG (default 300)
    """
    tex_path = Path(tex_file)
    output_path = Path(output_png)
    
    if not tex_path.exists():
        print(f"Error: {tex_file} not found")
        return False
    
    # Try pdflatex first, fall back to lualatex
    latex_cmd = None
    for cmd in ['pdflatex', 'lualatex']:
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            latex_cmd = cmd
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if not latex_cmd:
        print("Error: No LaTeX compiler found (pdflatex or lualatex)")
        print("Please install TeX Live or MiKTeX")
        return False
    
    # Change to directory containing tex file
    original_dir = os.getcwd()
    tex_dir = tex_path.parent
    tex_name = tex_path.stem
    
    try:
        os.chdir(tex_dir)
        
        # Compile LaTeX to PDF (run twice for proper references)
        print(f"Compiling {tex_file} with {latex_cmd}...")
        compile_cmd = [
            latex_cmd,
            '-interaction=nonstopmode',
            '-shell-escape',
            tex_path.name
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"LaTeX compilation failed:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # Run again for proper references
        subprocess.run(compile_cmd, capture_output=True, text=True)
        
        # Check if PDF was created
        pdf_path = tex_dir / f"{tex_name}.pdf"
        if not pdf_path.exists():
            print(f"Error: PDF not created: {pdf_path}")
            return False
        
        # Convert PDF to PNG using pdf2image or ImageMagick
        try:
            from pdf2image import convert_from_path
            print(f"Converting PDF to PNG (DPI={dpi})...")
            images = convert_from_path(pdf_path, dpi=dpi)
            if images:
                images[0].save(output_path, 'PNG')
                print(f"Success! Image saved to: {output_path}")
                return True
        except ImportError:
            print("pdf2image not found. Trying ImageMagick...")
            try:
                # Try ImageMagick convert command
                subprocess.run([
                    'magick', 'convert',
                    '-density', str(dpi),
                    pdf_path,
                    output_path
                ], check=True, capture_output=True)
                print(f"Success! Image saved to: {output_path}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Error: Neither pdf2image nor ImageMagick found")
                print("Install with: pip install pdf2image")
                print("Or install ImageMagick from https://imagemagick.org")
                return False
        
    finally:
        os.chdir(original_dir)
        # Clean up auxiliary files
        for ext in ['.aux', '.log', '.pdf']:
            aux_file = tex_dir / f"{tex_name}{ext}"
            if aux_file.exists() and ext != '.pdf':  # Keep PDF for reference
                try:
                    aux_file.unlink()
                except:
                    pass
    
    return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_tikz.py <tex_file> [output_png] [dpi]")
        print("Example: python render_tikz.py diagram.tex diagram.png 300")
        sys.exit(1)
    
    tex_file = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else tex_file.replace('.tex', '.png')
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    success = render_tikz_to_png(tex_file, output_png, dpi)
    sys.exit(0 if success else 1)
