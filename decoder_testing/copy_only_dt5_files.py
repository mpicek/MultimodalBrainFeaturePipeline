import argparse
import os
import subprocess

def copy_file_with_rsync(source, destination):
    """
    Copies a file from source to destination using rsync.
    
    :param source: Path of the source file.
    :param destination: Path of the destination directory.
    """
    subprocess.run(["rsync", "-av", "--partial", "--progress", source, destination], check=True)

def main(source_dir, dest_dir):
    """
    Iterates over files in the source directory tree. If a file ends with .dt5,
    it is copied to the destination directory maintaining the full directory structure.
    
    :param source_dir: Root directory to search for .dt5 files.
    :param dest_dir: Root directory where .dt5 files will be copied.
    """
    for dir in os.listdir(source_dir):
        if not "2023_12" in dir:# and not "2023_10_" in dir:
            continue

        relevant_dir = os.path.join(source_dir, dir)

        for root, dirs, files in os.walk(relevant_dir):
            for file in files:
                if file.endswith(".dt5"):
                    source_file_path = os.path.join(root, file)

                    # Create the corresponding destination directory structure
                    relative_path = os.path.relpath(root, source_dir)
                    destination_dir = os.path.join(dest_dir, relative_path)
                    print("the destination:")
                    print(destination_dir)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)

                    # Copy the file
                    destination_file_path = os.path.join(destination_dir, file)
                    print(f"Copying: {source_file_path}\n To: {destination_file_path}")
                    copy_file_with_rsync(source_file_path, destination_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy .dt5 files maintaining the original directory structure.")
    parser.add_argument("source_dir", type=str, help="Source directory to search for .dt5 files.")
    parser.add_argument("dest_dir", type=str, help="Destination directory where .dt5 files will be copied.")
    args = parser.parse_args()
    
    main(args.source_dir, args.dest_dir)