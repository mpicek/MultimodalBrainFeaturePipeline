"""Interactive LED-ROI labeling, with nr-data finding the videos.

Same labeling as extract_LED_position.py -- same GUI, same two .npy artifacts per
recording -- but the videos are discovered through nr-data instead of walking a
folder, and the artifacts land in the session's own annotation folder:

    <01_ANNOTATION>/<subject>/<session>/led_position/
        <video_pts_clock_id>_LED_position.npy
        <video_pts_clock_id>_LED_binary_mask.npy

The file stem is the recording's nr-data video-pts clock id, which is what the
LED intensity parquet will carry in its `clock` column.
"""

import argparse
import os
from pathlib import Path

import av
import numpy as np
from nr_data.trial_config.TrialSession import TrialSession
from nr_data.video.video_utils import find_video_recordings

from LED_GUI_cropper import ImageCropper
from LED_utils import LED_ERROR_NO_LED, led_annotation_folder, resolve_sessions, save_LED_error

DOWNSCALE_FACTOR = 2
FRAMES_TO_SKIP_AT_START = 100  # skip the first few frames, which are often under-exposed/unstable
LED_POSITION_FOLDER_NAME = "led_position"


def grab_frame_for_cropping(video_path, skip_frames=FRAMES_TO_SKIP_AT_START):
    """
    Decode one frame from partway into the video, to select the LED's ROI on.

    Decoded sequentially rather than seeking: a frame-index seek is unreliable on H.264 in
    Matroska, and reading `skip_frames` frames of a file that is opened anyway costs little.
    A video shorter than `skip_frames` yields its first frame, as its whole point is to give the
    user something to draw on. PyAV rather than cv2 so the box is drawn on exactly the pixels
    get_LED_signal_nr_data.py later measures.

    Raises RuntimeError when no frame can be decoded at all. That is deliberately not recorded as
    "no LED": such a failure is often transient (an unmounted network drive), so the caller should
    leave the recording unlabeled and re-offer it on the next run rather than recording a
    permanent verdict about it.
    """
    first_frame = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            image = frame.to_ndarray(format="bgr24")
            if index >= skip_frames:
                return image
            if first_frame is None:
                first_frame = image

    if first_frame is None:
        raise RuntimeError(f"Could not decode any frame from {video_path}")
    return first_frame


def label_recording(meta, led_position_folder):
    """Label one recording's LED ROI, writing the position and the mask (or a no-LED marker)."""
    position_path = led_position_folder / f"{meta.video_pts_clock_id}_LED_position.npy"
    binary_mask_path = led_position_folder / f"{meta.video_pts_clock_id}_LED_binary_mask.npy"

    frame = grab_frame_for_cropping(meta.video_path)
    cropped_LED_image_colorful, ref_point = ImageCropper(frame).show_and_crop_image()

    if ref_point is None:
        # the user pressed 'n' -- record that this video has no LED, so that it isn't
        # offered for labeling again on every subsequent run of this script
        print(f"  -> {LED_ERROR_NO_LED}")
        save_LED_error(position_path, LED_ERROR_NO_LED)
        save_LED_error(binary_mask_path, LED_ERROR_NO_LED)
        return

    mask_shape = cropped_LED_image_colorful[..., 2][::DOWNSCALE_FACTOR, ::DOWNSCALE_FACTOR].shape
    binary_mask = np.full(mask_shape, True)

    np.save(position_path, ref_point)
    np.save(binary_mask_path, binary_mask)


def extract_led_position_session(session, dry_run=False):
    """Label every unlabeled recording of one session. Returns (recordings, labeled)."""
    metas = find_video_recordings(session)
    videos = [meta for meta in metas if meta.video_path is not None]
    for meta in metas:
        if meta.video_path is None:
            print(f"  No video file for {meta.video_pts_clock_id} ({meta.csv_path.name}). Skipping.")
    if not videos:
        print("  No video recordings found.")
        return 0, 0

    led_position_folder = led_annotation_folder(session, LED_POSITION_FOLDER_NAME, create=not dry_run)

    labeled = 0
    for meta in videos:
        position_path = led_position_folder / f"{meta.video_pts_clock_id}_LED_position.npy"
        if position_path.exists():
            print(f"  {meta.video_pts_clock_id} already labeled. Skipping.")
            continue

        if dry_run:
            print(f"  would label {meta.video_pts_clock_id} ({meta.video_path.name})")
            labeled += 1
            continue

        print(f"  Labeling {meta.video_pts_clock_id} ({meta.video_path})")
        try:
            label_recording(meta, led_position_folder)
            labeled += 1
        except Exception as e:
            # nothing is written, so the recording stays unlabeled and gets re-offered next run
            print(f"  Error with {meta.video_path}: {e}")

    return len(videos), labeled


def main(identifiers, projects_folder=None, dry_run=False):
    if projects_folder is not None:
        # nr-data reads PROJECTS_FOLDER from the environment and never loads a .env itself
        os.environ["PROJECTS_FOLDER"] = str(Path(projects_folder).expanduser())

    sessions = resolve_sessions(identifiers)

    print(f"{'Would label' if dry_run else 'Labeling'} LED positions for {len(sessions)} session(s):")
    for session in sessions:
        print(f"  {session}")
    print(f"In total {len(sessions)} session(s).")

    total_recordings = total_labeled = 0
    for name in sessions:
        session = TrialSession.parse_argument(name)
        print(f"Session {name}:")
        recordings, labeled = extract_led_position_session(session, dry_run=dry_run)
        total_recordings += recordings
        total_labeled += labeled

    verb = "would label" if dry_run else "labeled"
    print(f"Finished: {verb} {total_labeled} of {total_recordings} recording(s) in {len(sessions)} session(s).")

    if identifiers and total_recordings == 0:
        raise SystemExit(f"No video recording found for any of: {', '.join(identifiers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label the LED position in every video nr-data finds.")
    parser.add_argument(
        "identifier",
        nargs="*",
        help="Trial, patient and/or session identifiers to process (default: every session).",
    )
    parser.add_argument(
        "--projects-folder",
        help="Projects folder to read (default: the PROJECTS_FOLDER environment variable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be labeled and exit, without writing a file or opening a window.",
    )
    args = parser.parse_args()

    main(args.identifier, projects_folder=args.projects_folder, dry_run=args.dry_run)
