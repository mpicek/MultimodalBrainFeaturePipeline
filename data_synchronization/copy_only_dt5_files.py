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
    it is copied to the destination directory maintaining the last two parent directories structure.
    
    :param source_dir: Root directory to search for .dt5 files.
    :param dest_dir: Root directory where .dt5 files will be copied.
    """
    for dir in os.listdir(source_dir):
        if not "2023_12_" in dir:# and not "2023_10_" in dir:
            continue

        relevant_dir = os.path.join(source_dir, dir)

        for root, dirs, files in os.walk(relevant_dir):
            for file in files:
                if file.endswith(".dt5"):
                    source_file_path = os.path.join(root, file)

                    # continue only if the basename includes "2023_12_*"
                    # path_without_two_parents = os.path.dirname(os.path.dirname(source_file_path))
                    # print(source_file_path)
                    # print(path_without_two_parents)

                    # new_path = os.path.join(dest_dir, source_file_path[len(path_without_two_parents) + 1:])
                    # print(new_path)
                    new_dest_subdir = os.path.join(dest_dir, dir)
                    if not os.path.exists(new_dest_subdir):
                        os.makedirs(new_dest_subdir)

                    # Extract the 'parent/next_parent' structure
                    print(f"Copying: {source_file_path}", "\n To: ", os.path.join(new_dest_subdir, file))
                    copy_file_with_rsync(source_file_path, os.path.join(new_dest_subdir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy .mat files maintaining part of their original directory structure.")
    parser.add_argument("source_dir", type=str, help="Source directory to search for .mat files.")
    parser.add_argument("dest_dir", type=str, help="Destination directory where .mat files will be copied.")
    args = parser.parse_args()
    
    main(args.source_dir, args.dest_dir)