"""
Script này đã được tích hợp vào prepare_data.py.
Giữ lại để backward compatibility nhưng sẽ redirect.
"""
import sys
from pathlib import Path

print("[INFO] extract_embedding.py is deprecated.")
print("[INFO] Please use prepare_data.py instead, which includes embedding extraction.")
print("[INFO] Running prepare_data.py...")

# Import và chạy prepare_data
from scripts.prepare_data import main

if __name__ == "__main__":
    main()
