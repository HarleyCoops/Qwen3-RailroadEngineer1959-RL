import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Railroad Extraction Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    args = parser.parse_args()
    
    print(f"Processing {args.pdf}...")

if __name__ == "__main__":
    main()
