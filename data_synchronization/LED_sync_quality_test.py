import os
import argparse
import pandas as pd
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

def display_images(video_name, ecog_name, sync_images_path, max_width, max_height):
    video_name = video_name[:-4]
    image_names = [
        video_name + "_" + ecog_name + "_corr.png",
        video_name + "_" + ecog_name + "_whole.png",
        video_name + "_" + ecog_name + "_beginning.png",
        video_name + "_" + ecog_name + "_ending.png",
        video_name + "_" + ecog_name + "_two_mins.png",
        video_name + "_" + ecog_name + "_last_two_mins.png",
    ]

    images = []
    image_paths = []
    for image_name in image_names:
        image_path = os.path.join(sync_images_path, image_name)
        if os.path.exists(image_path):
            print(image_path)
            image = Image.open(image_path)
            image.thumbnail((max_width, max_height))
            images.append(ImageTk.PhotoImage(image))
            image_paths.append(image_path)
        else:
            raise FileNotFoundError(f"Image {image_path} not found")

    return images, image_paths

def update_quality_status(video_name, status):
    df.loc[df['video_name'] == video_name, 'passed_quality_test'] = status
    df.to_csv(args.csv_log_table, index=False)

def key_pressed(event):
    if event.keysym == "Right":
        update_quality_status(video_name_entry.get(), 1)
    elif event.keysym == "Left":
        update_quality_status(video_name_entry.get(), 0)
    next_image()

def open_zoom(idx):
    global zoomed
    if idx >= len(current_image_paths):
        return
    image = Image.open(current_image_paths[idx])
    image.thumbnail((ZOOM_MAX_WIDTH, ZOOM_MAX_HEIGHT))
    zoom_photo = ImageTk.PhotoImage(image)
    zoom_label.config(image=zoom_photo)
    zoom_label.image = zoom_photo  # keep reference to prevent garbage collection

    for label in img_labels:
        label.grid_remove()
    video_name_entry.grid_remove()
    zoom_label.grid(row=0, column=0, rowspan=GRID_ROWS, columnspan=GRID_COLS, padx=3, pady=3, sticky="nsew")
    zoomed = True

def close_zoom():
    global zoomed
    zoom_label.grid_remove()
    for i, label in enumerate(img_labels):
        label.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=3, pady=3, sticky="nsew")
    video_name_entry.grid(row=GRID_ROWS, column=0, columnspan=GRID_COLS, padx=5, pady=5, sticky="ew")
    zoomed = False

def next_image():
    global index, current_image_paths
    index += 1
    if index < len(video_names):
        video_name = video_names[index]
        ecog_name = ecog_names[index]
        print(ecog_name)
        # if column "sync_failed" is 1, then skip this video (find the row based on video_name)
        while (
            df[df["video_name"] == video_name]["sync_failed"].values[0] == 1
        ) or ecog_name == "":
            update_quality_status(video_name, 0)
            index += 1
            if index >= len(video_names):
                root.quit()
                return
            video_name = video_names[index]
            ecog_name = ecog_names[index]
            print(ecog_name)

        if zoomed:
            close_zoom()

        images, current_image_paths = display_images(video_name, ecog_name, args.sync_images, max_width=MAX_IMG_WIDTH, max_height=MAX_IMG_HEIGHT)
        for i, img_label in enumerate(img_labels):
            img_label.config(image=images[i])
            img_label.image = images[i]  # Keep reference to prevent garbage collection
        video_name_entry.delete(0, tk.END)
        video_name_entry.insert(0, video_name)
        root.update()
    else:
        root.quit()

def create_quality_column_if_not_exists(df):
    if "passed_quality_test" not in df.columns:
        df["passed_quality_test"] = None

def main():

    global index
    index = -1
    next_image()

    root.bind("<KeyPress>", key_pressed)
    root.mainloop()

if __name__ == "__main__":
    """
    PRESS RIGHT ARROW KEY to mark the video as passed quality test
    PRESS LEFT ARROW KEY to mark the video as failed quality test
    """
    parser = argparse.ArgumentParser(description="Manual quality control GUI app")
    parser.add_argument("csv_log_table", help="Path to the CSV log table")
    parser.add_argument("sync_images", help="Path to the directory containing synchronized images")
    args = parser.parse_args()

    print("Press RIGHT ARROW KEY to mark the video as passed quality test")
    print("Press LEFT ARROW KEY to mark the video as failed quality test")
    print("Click on any image to zoom in, click again to zoom out")

    df = pd.read_csv(args.csv_log_table)
    create_quality_column_if_not_exists(df)

    video_names = df[df["passed_quality_test"].isnull()]["video_name"].tolist()
    ecog_paths = df[df["passed_quality_test"].isnull()]["path_ecog"].tolist()

    ecog_names = [
        os.path.basename(str(Path(n).parent)) if n == n else "" for n in ecog_paths
    ]

    root = tk.Tk()
    root.title("Manual Quality Control")

    # 6 images (corr, whole, beginning, ending, first two mins, last two mins) laid out
    # 2 columns x 3 rows. Size each thumbnail from the actual screen resolution instead of a
    # fixed pixel size, so the window fits on screen regardless of monitor size.
    GRID_COLS, GRID_ROWS = 2, 3
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    MAX_IMG_WIDTH = int(screen_w * 0.95 / GRID_COLS) - 20
    MAX_IMG_HEIGHT = int(screen_h * 0.85 / GRID_ROWS) - 20
    # click-to-zoom target size: near-fullscreen, used for whichever single image is enlarged
    ZOOM_MAX_WIDTH = int(screen_w * 0.95)
    ZOOM_MAX_HEIGHT = int(screen_h * 0.90)

    zoomed = False
    current_image_paths = []

    img_labels = [tk.Label(root, cursor="hand2") for _ in range(GRID_COLS * GRID_ROWS)]
    for i, label in enumerate(img_labels):
        label.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=3, pady=3, sticky="nsew")  # Use sticky to fill the label
        label.bind("<Button-1>", lambda event, idx=i: open_zoom(idx))

    zoom_label = tk.Label(root, cursor="hand2")
    zoom_label.bind("<Button-1>", lambda event: close_zoom())

    video_name_entry = tk.Entry(root)
    video_name_entry.grid(row=GRID_ROWS, column=0, columnspan=GRID_COLS, padx=5, pady=5, sticky="ew")

    main()