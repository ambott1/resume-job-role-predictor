"""
Utility functions for saving and loading objects, and other helper functions.
"""
import os
import sys
import joblib
from pathlib import Path


def save_object(file_path, obj):
    """
    Save an object to a file using joblib.
    
    Args:
        file_path (str): Path where the object should be saved
        obj: Object to save
        
    Raises:
        Exception: If there's an error saving the object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj, file_path)
        print(f"Object saved successfully to {file_path}")
        
    except Exception as e:
        raise Exception(f"Error occurred while saving object to {file_path}: {str(e)}")


def load_object(file_path):
    """
    Load an object from a file using joblib.
    
    Args:
        file_path (str): Path to the saved object
        
    Returns:
        Loaded object
        
    Raises:
        Exception: If there's an error loading the object
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        obj = joblib.load(file_path)
        print(f"Object loaded successfully from {file_path}")
        return obj
        
    except Exception as e:
        raise Exception(f"Error occurred while loading object from {file_path}: {str(e)}")


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent.parent


def get_artifacts_path():
    """
    Get the artifacts directory path.
    
    Returns:
        Path: Artifacts directory path
    """
    return get_project_root() / "artifacts"


def get_datasets_path():
    """
    Get the datasets directory path.
    
    Returns:
        Path: Datasets directory path
    """
    return get_project_root() / "datasets"



