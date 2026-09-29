"""Wisci ECoG aligned onto a session's video recordings, written as one parquet.

Session-wide version of nr-data's examples/load_wisci_aligned_to_kin_cameras.py: for every
(wisci recording, video recording) pair the session's alignment graph connects, the wisci samples
filmed by that video get the video's presentation timestamp (pts). The route is whatever the graph
offers - usually wisci -> camera wallclock (LED fit, wisci_to_cam) -> video pts (video_to_cam) - and
its uncertainty, the mapping.json ``uncertainty`` of every hop combined in quadrature, must be
within MAX_UNCERTAINTY_S for the pair to be used.

Output columns:

    timestamp | clock | sample_index | trigger_v | E00 ... E77     the wisci columns, unchanged
    <camera>_timestamp                                             pts in that camera's video, s
    <camera>_clock                                                 video_<camera>_<recording>_pts

and two kinds of rows, sorted by clock then timestamp:

    ECoG rows         one per wisci sample; a camera's columns are null where it was not filming
    video-only rows   one per frame of an accepted video that no ECoG sample is concurrent with
                      (filming started before or ran past the recording). The frame is placed on
                      the nearest aligned recording's clock; the ECoG columns are null, so a null
                      sample_index marks these rows.

By default both are kept in full. --drop-ecog-without-video drops ECoG rows no video filmed,
--drop-video-without-ecog drops the video-only rows; with both, only ECoG filmed by a video is left.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from nr_data.alignment.alignment_utils import (
    AlignmentPath,
    AlignmentStep,
    _mapping_graph,
    datetime_expr_to_unit,
    resolve_alignment_path,
    timestamp_alignment_path_expr,
    unit_expr_to_zurich_datetime,
)
from nr_data.ecog.layouts import find_wisci_recording_folders, wisci_clock_id
from nr_data.ecog.wisci_utils import load_wisci_recording
from nr_data.trial_config.TrialSession import TrialSession
from nr_data.utils import ZURICH_TIME_ZONE
from nr_data.video.video_utils import VideoRecordingMeta, find_video_recordings, read_video_timestamps

# Largest accepted alignment uncertainty, in seconds. mapping.json stores each hop's residual std
# (the LED fit's ~7 ms, the video fit's sub-ms); 50 ms is 2.5 frames at 50 fps, 1.5 at 30 fps.
MAX_UNCERTAINTY_S = 0.05

WISCI_TIMESTAMP_DTYPE = pl.Datetime("ns", ZURICH_TIME_ZONE)


@dataclass(frozen=True)
class Pair:
    """One wisci recording reaching one video, and the stretch of it that video filmed."""

    wisci_clock: str
    video: VideoRecordingMeta
    path: AlignmentPath
    recording_span_s: tuple[float, float]  # the whole recording on its own clock, unix s
    window_s: tuple[float, float]  # the filmed part of it, unix s

    @property
    def uncertainty_s(self):
        return self.path.uncertainty / 1000.0  # AlignmentPath reports milliseconds

    @property
    def overlap_s(self):
        return max(0.0, self.window_s[1] - self.window_s[0])

    @property
    def route(self):
        hops = [self.path.source_clock, *(step.target_clock for step in self.path.steps)]
        return "GDP" if any(hop.startswith("gdp_") for hop in hops) else "LED"


def reversed_path(path):
    """The same route walked backwards, so a window and the samples inside it share one set of hops."""
    steps = tuple(
        AlignmentStep(step.target_clock, step.source_clock, step.mapping, not step.inverse)
        for step in reversed(path.steps)
    )
    return AlignmentPath(path.target_clock, path.source_clock, steps)


def map_seconds(values_s, path):
    """Map seconds on the path's source clock to seconds on its target clock."""
    expr = timestamp_alignment_path_expr(unit_expr_to_zurich_datetime(pl.col("t"), "s"), path)
    return pl.DataFrame({"t": values_s}).select(datetime_expr_to_unit(expr, "s")).to_series().to_list()


def video_frame_pts_s(video):
    """Every frame's pts of a video, or None when its csv carries no usable pts."""
    pts = read_video_timestamps(video.csv_path).frames["pts_s"].drop_nulls().to_numpy()
    return pts if pts.size else None


def wisci_span_s(frame):
    span = frame.select(datetime_expr_to_unit(pl.col("timestamp"), "s").alias("t")).select(
        pl.col("t").min().alias("start"), pl.col("t").max().alias("end")
    )
    row = span.collect().row(0)
    return float(row[0]), float(row[1])


def make_pair(wisci_clock, video, path, frame_pts, recording_span):
    """Place the video's pts span on the wisci clock and clip it to the recording.

    Mapping the video's ends backwards (rather than every sample forwards and filtering on pts)
    keeps samples outside a piecewise map's range out: np.interp would clamp them onto an end
    frame instead.
    """
    start, end = sorted(map_seconds([float(frame_pts[0]), float(frame_pts[-1])], reversed_path(path)))
    window = (max(start, recording_span[0]), min(end, recording_span[1]))
    return Pair(wisci_clock, video, path, recording_span, window)


def with_video_columns(frame, pairs):
    """The recording's samples with per-camera pts/clock columns, null where a camera was not filming."""
    wisci_s = datetime_expr_to_unit(pl.col("timestamp"), "s")
    columns = []
    for camera in sorted({pair.video.camera for pair in pairs}):
        timestamp_expr = clock_expr = None
        for pair in (pair for pair in pairs if pair.video.camera == camera):
            filmed = wisci_s.is_between(*pair.window_s)
            pts = datetime_expr_to_unit(timestamp_alignment_path_expr(pl.col("timestamp"), pair.path), "s")
            clock = pl.lit(pair.video.video_pts_clock_id)
            if timestamp_expr is None:
                timestamp_expr, clock_expr = pl.when(filmed).then(pts), pl.when(filmed).then(clock)
            else:
                timestamp_expr, clock_expr = timestamp_expr.when(filmed).then(pts), clock_expr.when(filmed).then(clock)
        columns.append(timestamp_expr.otherwise(None).alias(f"{camera}_timestamp"))
        columns.append(clock_expr.otherwise(None).alias(f"{camera}_clock"))
    return frame.with_columns(columns)


def filmed_only(frame, pairs):
    cameras = sorted({pair.video.camera for pair in pairs})
    return frame.filter(pl.any_horizontal(pl.col(f"{camera}_timestamp").is_not_null() for camera in cameras))


def video_only_rows(video, frame_pts, pairs):
    """The video's frames no ECoG sample is concurrent with, each on its nearest recording's clock.

    A frame inside any aligned recording is already represented by that recording's samples. One
    outside all of them goes to the recording closest to it in time, so a video spanning the gap
    between two recordings still yields each frame once.
    """
    timestamp_column, clock_column = f"{video.camera}_timestamp", f"{video.camera}_clock"
    spans = [sorted(map_seconds(list(pair.recording_span_s), pair.path)) for pair in pairs]
    distance = np.stack([np.maximum(start - frame_pts, 0) + np.maximum(frame_pts - end, 0) for start, end in spans])
    uncovered = distance.min(axis=0) > 0
    nearest = distance.argmin(axis=0)

    parts = []
    for index, pair in enumerate(pairs):
        pts = frame_pts[uncovered & (nearest == index)]
        if not pts.size:
            continue
        wisci_timestamp = timestamp_alignment_path_expr(
            unit_expr_to_zurich_datetime(pl.col(timestamp_column), "s"), reversed_path(pair.path)
        )
        parts.append(
            pl.DataFrame({timestamp_column: pts}).select(
                wisci_timestamp.cast(WISCI_TIMESTAMP_DTYPE).alias("timestamp"),
                pl.lit(pair.wisci_clock).alias("clock"),
                pl.col(timestamp_column),
                pl.lit(video.video_pts_clock_id).alias(clock_column),
            )
        )
    return parts


def main(
    session_name,
    projects_folder=None,
    output=None,
    max_uncertainty_s=MAX_UNCERTAINTY_S,
    drop_ecog_without_video=False,
    drop_video_without_ecog=False,
):
    if projects_folder is not None:
        # nr-data reads PROJECTS_FOLDER from the environment and never loads a .env itself
        os.environ["PROJECTS_FOLDER"] = str(Path(projects_folder).expanduser())
    output = Path(output) if output else Path.cwd() / f"{session_name}_wisci_aligned_to_video.parquet"

    session = TrialSession.parse_argument(session_name)
    videos = find_video_recordings(session)
    recordings = find_wisci_recording_folders(session)
    # Built once: resolve_alignment_path would otherwise re-read every mapping.json per pair.
    graph = _mapping_graph(session)
    print(f"{session.session}: {len(recordings)} wisci recording(s), {len(videos)} video(s).")

    frame_pts = {video.video_pts_clock_id: video_frame_pts_s(video) for video in videos}
    reachable, overlapping = set(), set()
    accepted: dict[str, list[Pair]] = {}  # video clock -> its accepted pairs
    ecog_parts = []
    synchronized = 0
    for _, nwb_path, timestamp, _ in recordings:
        wisci_clock = wisci_clock_id(timestamp)
        paths = {}
        for video in videos:
            path = resolve_alignment_path(session, wisci_clock, video.video_pts_clock_id, mapping_graph=graph)
            if path is not None and frame_pts[video.video_pts_clock_id] is not None:
                paths[video.video_pts_clock_id] = (video, path)
        reachable.update(paths)
        if not paths:
            print(f"  {wisci_clock}: no alignment to any video.")
            if drop_ecog_without_video:
                continue

        frame, _ = load_wisci_recording(nwb_path, clock_id=wisci_clock)
        recording_span = wisci_span_s(frame)
        pairs = [
            make_pair(wisci_clock, video, path, frame_pts[clock], recording_span)
            for clock, (video, path) in paths.items()
        ]
        pairs = [pair for pair in pairs if pair.overlap_s > 0]
        if paths and not pairs:
            print(f"  {wisci_clock}: aligned, but filmed by no video.")

        kept = []
        for pair in pairs:
            ok = pair.uncertainty_s <= max_uncertainty_s
            overlapping.add(pair.video.video_pts_clock_id)
            print(
                f"  {wisci_clock} -> {pair.video.video_pts_clock_id}: via {pair.route}, "
                f"{pair.uncertainty_s * 1000:.1f} ms, {pair.overlap_s:.0f} s filmed{'' if ok else '  REJECTED'}"
            )
            if ok:
                accepted.setdefault(pair.video.video_pts_clock_id, []).append(pair)
                kept.append(pair)

        if kept:
            synchronized += 1
            frame = with_video_columns(frame, kept)
            if drop_ecog_without_video:
                frame = filmed_only(frame, kept)
        elif drop_ecog_without_video:
            continue
        ecog_parts.append(frame.collect())

    video_parts = []
    if not drop_video_without_ecog:
        for video in videos:
            if video.video_pts_clock_id in accepted:
                video_parts += video_only_rows(video, frame_pts[video.video_pts_clock_id], accepted[video.video_pts_clock_id])

    print(f"Videos: {len(videos)}")
    print(f"  with an alignment path from a wisci recording: {len(reachable)}")
    print(f"  filming a wisci recording:                    {len(overlapping)}")
    print(f"  within {max_uncertainty_s * 1000:.0f} ms uncertainty:                  {len(accepted)}")
    print(f"Wisci recordings: {len(recordings)}, synchronized: {synchronized}")

    if not ecog_parts and not video_parts:
        raise SystemExit("Nothing to write: no wisci recording has an accepted video alignment.")
    df = pl.concat(ecog_parts + video_parts, how="diagonal_relaxed").sort(["clock", "timestamp"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    ecog_rows = sum(part.height for part in ecog_parts)
    print(f"Wrote {df.height} row(s) ({ecog_rows} ECoG, {df.height - ecog_rows} video-only), {df.width} column(s) to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a session's wisci ECoG aligned onto its videos' pts.")
    parser.add_argument("session", help="Trial session to process, e.g. UP2004_2025_11_11_EESMappingD2_BurstMapping25Hz.")
    parser.add_argument(
        "--projects-folder",
        help="Projects folder to read (default: the PROJECTS_FOLDER environment variable).",
    )
    parser.add_argument(
        "--output",
        help="Parquet file to write (default: ./<session>_wisci_aligned_to_video.parquet).",
    )
    parser.add_argument(
        "--max-uncertainty-s",
        type=float,
        default=MAX_UNCERTAINTY_S,
        help=f"Largest accepted alignment uncertainty in seconds (default: {MAX_UNCERTAINTY_S}).",
    )
    parser.add_argument(
        "--drop-ecog-without-video",
        action="store_true",
        help="Keep only ECoG samples filmed by an accepted video.",
    )
    parser.add_argument(
        "--drop-video-without-ecog",
        action="store_true",
        help="Leave out the frames of accepted videos that no ECoG sample is concurrent with.",
    )
    args = parser.parse_args()

    main(
        args.session,
        projects_folder=args.projects_folder,
        output=args.output,
        max_uncertainty_s=args.max_uncertainty_s,
        drop_ecog_without_video=args.drop_ecog_without_video,
        drop_video_without_ecog=args.drop_video_without_ecog,
    )
