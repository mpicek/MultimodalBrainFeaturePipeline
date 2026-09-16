"""LED intensity extraction into nr-data's canonical per-session parquet.

Measures the same thing as get_LED_signal.py -- the per-frame mean of the red channel inside
the labeled ROI -- but each sample now carries its frame's presentation timestamp (pts), and
the session's recordings are written as one canonical parquet:

    <02_PROCESSED>/<subject>/<session>/led_signals/
        data_layout.json
        runs/<timestamp>/
            data.parquet        timestamp (pts, Duration) | clock | led_intensity
            skipped_LED.json    the recordings that produced no rows, and why
            provenance.json, config.json, run.log, PROCESSING_COMPLETE

The ROI comes from <01_ANNOTATION>/<subject>/<session>/led_position/, written by
extract_LED_position_nr_data.py. A recording with no LED (or no ROI yet) contributes no rows:
its clock is then simply absent from LedSignalData(session).get_clocks(), so no reader can
mistake a placeholder for a measurement. The reason is recorded in run.log and skipped_LED.json.

Decoding is PyAV rather than cv2 because a sample is only useful with its pts, and the pts has
to come from the same decode as the pixels -- the acquisition CSV has one row fewer than the
container has frames, so pairing them by index would silently shift the trace by a frame.
"""

import argparse
import os
from pathlib import Path

import av
import numpy as np
import polars as pl
from nr_data.core.RunFolder import RunFolder
from nr_data.led_signal import LedSignalData, get_led_signal_folder
from nr_data.led_signal.layouts import LedSignalDataLayout
from nr_data.session_data import save_session_temporal_data
from nr_data.trial_config.TrialSession import TrialSession
from nr_data.utils import duration_s_expr
from nr_data.video.video_utils import find_video_recordings

from LED_utils import led_annotation_folder, load_LED_array, resolve_sessions

DOWNSCALE_FACTOR = 2
LED_POSITION_FOLDER_NAME = "led_position"
LED_SIGNAL_FOLDER_NAME = "led_signal"
SKIPPED_FILENAME = "skipped_LED.json"
SCRIPT_NAME = "get_LED_signal_nr_data"

# Why a recording produced no rows, as written into skipped_LED.json.
NO_VIDEO = "no video file"
NOT_LABELED = "no LED position/mask"
NO_LED = "annotated as having no LED"
FAILED = "extraction failed"


def extract_trace(video_path, led_position, binary_mask):
    """
    Decode one video once, returning (pts_s, intensity) -- one sample per frame.

    The measurement is the same as LED_utils.get_LED_signal_from_video: crop to the ROI, take the
    red channel, downscale, average the pixels the mask selects (0 when it selects none). Frames
    are requested as bgr24 so the red channel stays at index 2, exactly as it is under cv2.

    pts is emitted exactly as the container reports it and is never rebased: the acquisition
    timestamp CSVs are on that same timeline, so both sides agree frame for frame without having
    to agree on an origin.
    """
    binary_mask = binary_mask.astype(bool)
    pts_s = []
    intensity = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Frame {len(pts_s)} of {video_path} has no pts; it cannot be placed on a clock.")
            pts_s.append(float(frame.pts * stream.time_base))

            image = frame.to_ndarray(format="bgr24")
            cropped = image[led_position[0][1] : led_position[1][1], led_position[0][0] : led_position[1][0]]
            red = cropped[..., 2][::DOWNSCALE_FACTOR, ::DOWNSCALE_FACTOR]
            selected_pixels = red[binary_mask]
            intensity.append(float(np.mean(selected_pixels)) if selected_pixels.size > 0 else 0.0)

    return np.array(pts_s, dtype=np.float64), np.array(intensity, dtype=np.float64)


def load_cached_trace(trace_path):
    """(pts_s, intensity) from a cached trace, or None when there is nothing usable to reuse."""
    if not trace_path.exists():
        return None

    cached, error = load_LED_array(trace_path)
    if error is not None or cached.ndim != 2 or cached.shape[1] != 2:
        # an error marker, or an intensity-only array from before pts was recorded
        return None
    return cached[:, 0], cached[:, 1]


def trace_frame(clock, pts_s, intensity):
    """One recording's trace in the Table contract: timestamp (pts) | clock | led_intensity."""
    return pl.DataFrame({"pts_s": pts_s, "led_intensity": intensity}).select(
        duration_s_expr(pl.col("pts_s")).alias("timestamp"),
        pl.lit(clock).alias("clock"),
        pl.col("led_intensity").cast(pl.Float64),
    )


def session_is_completed(session):
    """Whether the session already has a completed LED run."""
    try:
        return LedSignalData(session).is_completed()
    except FileNotFoundError:
        # an earlier run failed before completing, so no run resolves yet
        return False


def recording_plan(session, dry_run=False):
    """
    What each of the session's recordings needs: (meta, trace_path, led_position, binary_mask).

    Recordings that cannot produce rows are returned as skip entries instead, each with its reason.
    The no-LED verdict is read here, before any run folder exists, so that a session where every
    recording is annotated as having no LED is recognised as having nothing to do rather than as a
    failed extraction.
    """
    led_position_folder = led_annotation_folder(session, LED_POSITION_FOLDER_NAME)
    led_signal_folder = led_annotation_folder(session, LED_SIGNAL_FOLDER_NAME, create=not dry_run)

    todo = []
    skipped = []
    for meta in find_video_recordings(session):
        entry = {"clock": meta.video_pts_clock_id, "camera": meta.camera, "recording": meta.recording}

        if meta.video_path is None:
            skipped.append({**entry, "reason": NO_VIDEO, "detail": str(meta.csv_path)})
            continue

        position_path = led_position_folder / f"{meta.video_pts_clock_id}_LED_position.npy"
        mask_path = led_position_folder / f"{meta.video_pts_clock_id}_LED_binary_mask.npy"
        if not position_path.exists() or not mask_path.exists():
            skipped.append({**entry, "reason": NOT_LABELED, "detail": str(led_position_folder)})
            continue

        try:
            led_position, position_error = load_LED_array(position_path)
            binary_mask, mask_error = load_LED_array(mask_path)
        except Exception as e:
            skipped.append({**entry, "reason": FAILED, "detail": f"unreadable annotation: {e}"})
            continue

        led_error = position_error or mask_error
        if led_error is not None:
            # labeling already established there is no usable LED here: no rows, and the verdict
            # stays where it was made, in led_position/
            skipped.append({**entry, "reason": NO_LED, "detail": led_error})
            continue

        trace_path = led_signal_folder / f"{meta.video_pts_clock_id}_LED_trace.npy"
        todo.append((meta, trace_path, led_position, binary_mask))

    return todo, skipped


def extract_session(session, dry_run=False, force=False):
    """
    Extract every labeled recording of one session into its canonical parquet.

    Returns (recordings, measured, already_complete): the last one separates a session that had
    nothing to do from one that had nothing to find.
    """
    if not force and session_is_completed(session):
        print("  Already has a completed LED run. Skipping (use --force to re-run).")
        return 0, 0, True

    todo, skipped = recording_plan(session, dry_run=dry_run)
    for entry in skipped:
        print(f"  {entry['clock']}: {entry['reason']}. Skipping.")

    if dry_run:
        for meta, trace_path, _, _ in todo:
            source = "cached trace" if load_cached_trace(trace_path) is not None else "decode"
            print(f"  would measure {meta.video_pts_clock_id} ({source})")
        return len(todo), len(todo), False

    if not todo:
        # Nothing here can produce a trace - every recording is unlabeled, annotated as having no
        # LED, or has no video. That is a legitimate end state for a session, not a failed run, so
        # no run folder is created and nothing is marked as an error.
        print("  Nothing to measure (no labeled recording with an LED). Skipping session.")
        return 0, 0, False

    config = {
        "downscale_factor": DOWNSCALE_FACTOR,
        "led_position_folder": LED_POSITION_FOLDER_NAME,
        "led_signal_cache_folder": LED_SIGNAL_FOLDER_NAME,
        "force": force,
    }
    # Created before extraction starts, so its provenance reflects when the work began.
    run_folder = RunFolder.create(get_led_signal_folder(session) / "runs", config=config, script_name=SCRIPT_NAME)

    measured = 0
    with run_folder.processing(
        error_content=f"Error while extracting LED signals for {session.session}.",
        interrupted_content=f"Interrupted while extracting LED signals for {session.session}.",
        logger_name=SCRIPT_NAME,
    ) as run:
        traces = []
        for meta, trace_path, led_position, binary_mask in todo:
            clock = meta.video_pts_clock_id
            try:
                cached = load_cached_trace(trace_path)
                if cached is not None:
                    pts_s, intensity = cached
                    run.log.info("%s: reusing %d cached sample(s).", clock, len(pts_s))
                else:
                    run.log.info("%s: decoding %s", clock, meta.video_path)
                    pts_s, intensity = extract_trace(meta.video_path, led_position, binary_mask)
                    np.save(trace_path, np.column_stack([pts_s, intensity]))
                    run.log.info("%s: %d sample(s), pts %.3f-%.3f s.", clock, len(pts_s), pts_s[0], pts_s[-1])
            except Exception as e:
                # one unreadable video must not cost the whole session
                run.log.warning("%s: %s. No rows written.", clock, e)
                skipped.append(
                    {
                        "clock": clock,
                        "camera": meta.camera,
                        "recording": meta.recording,
                        "reason": FAILED,
                        "detail": str(e),
                    }
                )
                continue

            traces.append(trace_frame(clock, pts_s, intensity))
            measured += 1

        if not traces:
            # Every recording that should have produced a trace failed to: a real error, so the
            # run is marked failed and stays invisible to readers.
            raise RuntimeError(f"No LED trace could be extracted for {session.session}")

        run_folder.write_json(SKIPPED_FILENAME, {"session": session.session, "skipped": skipped}, indent=4)

        # Sorted by timestamp because the reader promises polars the column is sorted
        # (session_data._load -> set_sorted); per-clock order is unaffected either way.
        save_session_temporal_data(
            session,
            LedSignalDataLayout.DATA_FOLDER_NAME,
            run_folder,
            pl.concat(traces).sort("timestamp"),
            complete_run=False,
        )

    print(f"  Wrote {measured} trace(s) to {run_folder.path}")
    return len(todo), measured, False


def main(identifiers, projects_folder=None, dry_run=False, force=False, raise_on_error=False):
    if projects_folder is not None:
        # nr-data reads PROJECTS_FOLDER from the environment and never loads a .env itself
        os.environ["PROJECTS_FOLDER"] = str(Path(projects_folder).expanduser())

    sessions = resolve_sessions(identifiers)
    print(f"{'Would extract' if dry_run else 'Extracting'} LED signals for {len(sessions)} session(s):")
    for session in sessions:
        print(f"  {session}")

    total_recordings = total_measured = total_complete = 0
    failed = []
    for name in sessions:
        session = TrialSession.parse_argument(name)
        print(f"Session {name}:")
        try:
            recordings, measured, already_complete = extract_session(session, dry_run=dry_run, force=force)
        except Exception as e:
            # One session's failure is reported and counted, never silently swallowed, but it does
            # not cost every session after it in the sweep. Its run folder is already marked
            # PROCESSING_ERROR and stays invisible to readers.
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))
            if raise_on_error:
                raise
            continue

        total_recordings += recordings
        total_measured += measured
        total_complete += int(already_complete)

    verb = "would measure" if dry_run else "measured"
    print(f"Finished: {verb} {total_measured} of {total_recordings} recording(s) in {len(sessions)} session(s).")

    if failed:
        print(f"{len(failed)} session(s) failed:")
        for name, error in failed:
            print(f"  {name}: {error}")
        raise SystemExit(1)

    # Nothing to do is not nothing to find: a session skipped because it is already complete
    # is a successful run, an identifier that matches no recording at all is not.
    if identifiers and total_recordings == 0 and total_complete == 0:
        raise SystemExit(f"No video recording found for any of: {', '.join(identifiers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the LED signal from every video nr-data finds.")
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
        help="List what would be processed and exit, without writing a file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run sessions that already have a completed LED run.",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Stop at the first failing session instead of reporting it and carrying on.",
    )
    args = parser.parse_args()

    main(
        args.identifier,
        projects_folder=args.projects_folder,
        dry_run=args.dry_run,
        force=args.force,
        raise_on_error=args.raise_on_error,
    )
