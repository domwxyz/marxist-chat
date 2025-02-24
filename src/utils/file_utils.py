import os
import shutil
from pathlib import Path
import json
from typing import Dict, List, Any, Union

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
    
def directory_has_content(directory_path: Union[str, Path], file_pattern: str = "*") -> bool:
    """Check if a directory exists and contains any files matching the pattern"""
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        return False
        
    # Check if there are any files matching the pattern
    return any(path.glob(file_pattern))

def delete_directory(directory_path: Union[str, Path]) -> bool:
    """Delete a directory and all its contents"""
    path = Path(directory_path)
    if not path.exists():
        return False
    
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"Error deleting directory {path}: {e}")
        return False

def get_file_count(directory_path: Union[str, Path], extension: str = None) -> int:
    """Count files in a directory, optionally filtering by extension"""
    path = Path(directory_path)
    if not path.exists():
        return 0
        
    if extension:
        return len(list(path.glob(f"*.{extension}")))
    else:
        return len([f for f in path.iterdir() if f.is_file()])

def get_file_size(file_path: Union[str, Path]) -> int:
    """Get the size of a file in bytes"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return 0
    
    return path.stat().st_size

def get_directory_size(directory_path: Union[str, Path]) -> int:
    """Get the total size of all files in a directory in bytes"""
    path = Path(directory_path)
    if not path.exists():
        return 0
        
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON data from a file"""
    path = Path(file_path)
    if not path.exists():
        return {}
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}")
        return {}

def save_json(data: Dict, file_path: Union[str, Path]) -> bool:
    """Save data to a JSON file"""
    path = Path(file_path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}")
        return False

def get_file_list(directory_path: Union[str, Path], extension: str = None) -> List[Path]:
    """Get a list of files in a directory, optionally filtering by extension"""
    path = Path(directory_path)
    if not path.exists():
        return []
        
    if extension:
        return list(path.glob(f"*.{extension}"))
    else:
        return [f for f in path.iterdir() if f.is_file()]
