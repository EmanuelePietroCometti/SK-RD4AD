import os

def clean_unmatched_files(folder_to_clean: str, reference_folder: str, dry_run: bool = True):
    """
    Deletes files in 'folder_to_clean' that do not have a corresponding 
    file (matching base name) in 'reference_folder'.
    
    Args:
        folder_to_clean (str): Path to the folder where files will be deleted (e.g., test/nodi).
        reference_folder (str): Path to the reference folder (e.g., ground_truth/nodi).
        dry_run (bool): If True, it only prints what would be deleted without actually deleting.
    """
    
    # Ensure both directories exist
    if not os.path.exists(folder_to_clean) or not os.path.exists(reference_folder):
        print("Error: One or both directories do not exist. Please check the paths.")
        return

    # Get a set of base names (without extensions) from the reference folder
    # Example: 'image01.png' becomes 'image01'
    reference_basenames = set()
    for filename in os.listdir(reference_folder):
        if os.path.isfile(os.path.join(reference_folder, filename)):
            basename, _ = os.path.splitext(filename)
            reference_basenames.add(basename)

    # Iterate through the folder to clean and check against the reference set
    deleted_count = 0
    for filename in os.listdir(folder_to_clean):
        filepath = os.path.join(folder_to_clean, filename)
        
        # Only process files, skip subdirectories
        if os.path.isfile(filepath):
            basename, _ = os.path.splitext(filename)
            
            # If the base name is not found in the reference folder, delete it
            if basename not in reference_basenames:
                if dry_run:
                    print(f"[DRY RUN] Would delete: {filepath}")
                else:
                    os.remove(filepath)
                    print(f"[DELETED] {filepath}")
                deleted_count += 1

    # Summary output
    status = "Simulated" if dry_run else "Actually deleted"
    print(f"\nDone. {status} {deleted_count} files.")

if __name__ == "__main__":
    # Define your paths here (use absolute paths or relative to where you run the script)
    # Example: We want to delete test images that don't have a mask
    TARGET_FOLDER = "mvtec/reda/test/paglie"
    REFERENCE_FOLDER = "mvtec/reda/ground_truth/paglie"
    
    clean_unmatched_files(TARGET_FOLDER, REFERENCE_FOLDER, dry_run=False)
    