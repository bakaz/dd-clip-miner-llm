from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import get_padding_config
from .ffmpeg import cut_audio, cut_video, get_duration
from .merger import _merge_adjacent_matches, _split_indices_by_time_gap, build_content_results
from .models import ContentMatch, ContentResult, TranscriptSegment
from .paths import safe_path_part
from .run_paths import (
    portable_run_dir,
    recorded_run_dir_from_context,
    resolve_input_video,
    resolve_run_path,
)


class PostMergeError(RuntimeError):
    """Raised when a drag-drop merge request cannot be mapped back to ASR data."""


def post_merge_from_context(
    context_path: str | Path,
    *files: str | Path,
) -> dict[str, Any]:
    if len(files) < 2:
        raise PostMergeError("post-merge requires at least two dragged files")

    context_file = Path(context_path)
    context = _load_json_object(context_file)
    recorded_run_dir = recorded_run_dir_from_context(context)
    run_dir = portable_run_dir(context_file.parent, recorded_run_dir)
    content_type = str(context.get("content_type") or "song")
    if content_type != "song":
        raise PostMergeError(f"post-merge currently supports song clips only, got: {content_type}")

    dragged_files = [_validate_dragged_file(path) for path in files]
    output_suffix = dragged_files[0].suffix.lower()
    if output_suffix not in {".mp4", ".mp3"}:
        raise PostMergeError(f"Unsupported file type: {dragged_files[0]}")
    for dragged_file in dragged_files[1:]:
        if dragged_file.suffix.lower() not in {".mp4", ".mp3"}:
            raise PostMergeError(f"Unsupported file type: {dragged_file}")
        if dragged_file.suffix.lower() != output_suffix:
            raise PostMergeError("Dragged files must have the same extension (.mp4 with .mp4, or .mp3 with .mp3)")

    config = _load_config_snapshot(context)
    transcript_path = _context_path(context, "transcript_path", run_dir=run_dir, recorded_run_dir=recorded_run_dir)
    matches_path = _context_path(context, "matches_path", run_dir=run_dir, recorded_run_dir=recorded_run_dir)
    reports_path = _context_path(context, "reports_path", run_dir=run_dir, recorded_run_dir=recorded_run_dir)

    segments = _load_transcript(transcript_path)
    matches = _load_matches(matches_path)
    report_results = _load_report_results(reports_path, content_type)
    if not report_results:
        raise PostMergeError(f"No report results found in: {reports_path}")

    merge_results = _find_report_results_for_paths(
        dragged_files,
        report_results,
        run_dir,
    )

    total_duration, source_video = _source_video_and_duration(
        context, run_dir, recorded_run_dir=recorded_run_dir,
    )
    source_by_index = _source_indices_by_result_index(segments, matches, total_duration, config, content_type)
    index_groups: list[list[int]] = []
    for merge_result in merge_results:
        indices = source_by_index.get(merge_result.index)
        if not indices:
            raise PostMergeError(f"Could not map report index {merge_result.index} back to ASR segments")
        index_groups.append(indices)

    merged_indices = list(
        range(
            min(group[0] for group in index_groups),
            max(group[-1] for group in index_groups) + 1,
        )
    )
    title, artist = _merged_title_artist_many(merge_results)
    merged_match = ContentMatch(
        content_type="song",
        title=title,
        artist=artist,
        segment_indices=merged_indices,
        confidence=max(result.confidence for result in merge_results),
        tags=sorted({tag for result in merge_results for tag in result.tags}),
        lyrics_snippet="",
    )
    merged_results = build_content_results(
        segments,
        [merged_match],
        total_duration,
        _force_single_song_config(config, total_duration),
        "song",
    )
    if not merged_results:
        raise PostMergeError("Merged ASR range was filtered out by current song duration settings")
    merged_result = merged_results[0]

    output_dir = dragged_files[0].parent
    base_stem = _merge_output_stem(dragged_files)

    output_config = config.get("output", {})
    video_codec = str(output_config.get("video_codec") or "copy")
    audio_bitrate_kbps = int(output_config.get("audio_bitrate_kbps") or 320)

    video_output: Path | None = None
    audio_output: Path | None = None
    if output_suffix == ".mp4":
        video_output = _unique_path(output_dir / f"{base_stem}.mp4")
        cut_video(source_video, video_output, merged_result.start, merged_result.end, video_codec=video_codec)
        merged_result.video_path = video_output
    else:
        audio_output = _unique_path(output_dir / f"{base_stem}.mp3")
        cut_audio(
            source_video,
            audio_output,
            merged_result.start,
            merged_result.end,
            copy_codec=False,
            bitrate_kbps=audio_bitrate_kbps,
        )
        merged_result.audio_path = audio_output

    return {
        "source_video": str(source_video),
        "video_path": str(video_output) if video_output else None,
        "audio_path": str(audio_output) if audio_output else None,
        "output_path": str(video_output or audio_output),
        "output_type": output_suffix.lstrip("."),
        "start": merged_result.start,
        "end": merged_result.end,
        "duration": merged_result.duration,
        "segment_indices": merged_indices,
        "title": merged_result.title,
        "artist": merged_result.artist,
    }


def _validate_dragged_file(value: str | Path) -> Path:
    path = Path(value)
    if not path.exists():
        raise PostMergeError(f"Dragged file not found: {path}")
    if not path.is_file():
        raise PostMergeError(f"Dragged path is not a file: {path}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PostMergeError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PostMergeError(f"Invalid JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise PostMergeError(f"Expected JSON object in: {path}")
    return data


def _load_json_list(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PostMergeError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PostMergeError(f"Invalid JSON file: {path}") from exc
    if not isinstance(data, list):
        raise PostMergeError(f"Expected JSON list in: {path}")
    return data


def _context_path(
    context: dict[str, Any],
    key: str,
    *,
    run_dir: Path,
    recorded_run_dir: Path | None = None,
) -> Path:
    value = context.get(key)
    if not value:
        raise PostMergeError(f"merge context is missing {key!r}")
    path = resolve_run_path(value, run_root=run_dir, recorded_run_dir=recorded_run_dir)
    if not path.exists():
        raise PostMergeError(f"Required file not found: {path}")
    return path


def _load_config_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    config = context.get("config")
    if not isinstance(config, dict):
        raise PostMergeError("merge context is missing config snapshot")
    return config


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in _load_json_list(path):
        if not isinstance(item, dict):
            continue
        segments.append(
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=str(item.get("text", "")),
            )
        )
    if not segments:
        raise PostMergeError(f"No transcript segments found in: {path}")
    return segments


def _load_matches(path: Path) -> list[ContentMatch]:
    matches: list[ContentMatch] = []
    for item in _load_json_list(path):
        if not isinstance(item, dict):
            continue
        matches.append(
            ContentMatch(
                content_type=str(item.get("content_type") or "song"),
                title=str(item.get("title") or "merged_song"),
                segment_indices=[int(i) for i in item.get("segment_indices", [])],
                confidence=float(item.get("confidence", 0.5)),
                tags=[str(t) for t in item.get("tags", [])],
                description=str(item.get("description") or ""),
                artist=str(item.get("artist") or ""),
                lyrics_snippet=str(item.get("lyrics_snippet") or ""),
            )
        )
    if not matches:
        raise PostMergeError(f"No matches found in: {path}")
    return matches


def _load_report_results(path: Path, content_type: str) -> list[ContentResult]:
    results: list[ContentResult] = []
    for item in _load_json_list(path):
        if not isinstance(item, dict):
            continue
        results.append(
            ContentResult(
                index=int(item.get("index") or len(results) + 1),
                content_type=str(item.get("content_type") or content_type),
                title=str(item.get("title") or f"{content_type}_{len(results) + 1:03d}"),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                duration=float(item.get("duration", 0.0)),
                transcript=str(item.get("transcript") or ""),
                confidence=float(item.get("confidence", 0.5)),
                tags=[str(t) for t in item.get("tags", [])],
                description=str(item.get("description") or ""),
                artist=str(item.get("artist") or ""),
                audio_path=Path(item["audio_path"]) if item.get("audio_path") else None,
                video_path=Path(item["video_path"]) if item.get("video_path") else None,
                errors=[str(e) for e in item.get("errors", [])],
            )
        )
    return results


def _find_report_result(path: Path, results: list[ContentResult], run_dir: Path) -> ContentResult:
    needle_keys = _path_keys(path, run_dir)
    for result in results:
        for candidate in (result.video_path, result.audio_path):
            if candidate is not None and needle_keys & _path_keys(candidate, run_dir):
                return result
    export_sequence = _export_sequence_number(path)
    if export_sequence is not None:
        for result in results:
            if _result_export_sequence(result, path.suffix) == export_sequence:
                return result
    suffix_index = _export_index_suffix(path)
    if suffix_index is not None:
        for result in results:
            if result.index == suffix_index and _result_has_extension(result, path.suffix):
                return result
    raise PostMergeError(f"Dragged file is not listed in the song report: {path}")


def _find_report_results_for_paths(
    paths: list[Path],
    results: list[ContentResult],
    run_dir: Path,
) -> list[ContentResult]:
    resolved: list[ContentResult | None] = [None] * len(paths)
    errors: list[PostMergeError | None] = [None] * len(paths)

    for index, path in enumerate(paths):
        try:
            resolved[index] = _find_report_result(path, results, run_dir)
        except PostMergeError as exc:
            errors[index] = exc

    changed = True
    while changed:
        changed = False
        known = [item for item in resolved if item is not None]
        used_indexes = {item.index for item in known}
        for index, path in enumerate(paths):
            if resolved[index] is not None:
                continue
            for companion in known:
                candidate = _infer_unsuffixed_companion(path, companion, results)
                if candidate is not None and candidate.index not in used_indexes:
                    resolved[index] = candidate
                    used_indexes.add(candidate.index)
                    changed = True
                    break

    if any(item is None for item in resolved):
        first_missing = next(index for index, item in enumerate(resolved) if item is None)
        raise errors[first_missing] or PostMergeError(
            f"Dragged file is not listed in the song report: {paths[first_missing]}"
        )
    return [resolved[index] for index in range(len(paths))]


def _find_report_results(
    first_path: Path,
    second_path: Path,
    results: list[ContentResult],
    run_dir: Path,
) -> tuple[ContentResult, ContentResult]:
    first_error: PostMergeError | None = None
    second_error: PostMergeError | None = None
    first: ContentResult | None = None
    second: ContentResult | None = None
    try:
        first = _find_report_result(first_path, results, run_dir)
    except PostMergeError as exc:
        first_error = exc
    try:
        second = _find_report_result(second_path, results, run_dir)
    except PostMergeError as exc:
        second_error = exc

    if first is None and second is not None:
        first = _infer_unsuffixed_companion(first_path, second, results)
    if second is None and first is not None:
        second = _infer_unsuffixed_companion(second_path, first, results)

    if first is None:
        raise first_error or PostMergeError(f"Dragged file is not listed in the song report: {first_path}")
    if second is None:
        raise second_error or PostMergeError(f"Dragged file is not listed in the song report: {second_path}")
    return first, second


def _export_index_suffix(path: Path) -> int | None:
    import re

    match = re.search(r"_(\d{3})$", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _export_sequence_number(path: Path) -> int | None:
    import re

    match = re.search(r"】\s*(\d{3})\s*-", path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|[_-])(\d{3})-", path.stem)
    if match:
        return int(match.group(1))
    return None


def _result_export_sequence(result: ContentResult, suffix: str) -> int | None:
    suffix = suffix.lower()
    for candidate in (result.video_path, result.audio_path):
        if candidate is not None and candidate.suffix.lower() == suffix:
            sequence = _export_sequence_number(candidate)
            if sequence is not None:
                return sequence
    return None


def _result_has_extension(result: ContentResult, suffix: str) -> bool:
    suffix = suffix.lower()
    return any(
        candidate is not None and candidate.suffix.lower() == suffix
        for candidate in (result.video_path, result.audio_path)
    )


def _infer_unsuffixed_companion(
    path: Path,
    companion: ContentResult,
    results: list[ContentResult],
) -> ContentResult | None:
    if _export_index_suffix(path) is not None:
        return None
    candidates = [
        result
        for result in results
        if result.index != companion.index
        and result.title == companion.title
        and result.artist == companion.artist
        and _result_has_extension(result, path.suffix)
        and _result_export_index_suffix(result, path.suffix) is None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda result: abs(result.index - companion.index))


def _result_export_index_suffix(result: ContentResult, suffix: str) -> int | None:
    suffix = suffix.lower()
    for candidate in (result.video_path, result.audio_path):
        if candidate is not None and candidate.suffix.lower() == suffix:
            return _export_index_suffix(candidate)
    return None


def _path_keys(path: Path, run_dir: Path) -> set[str]:
    candidates = [path]
    if not path.is_absolute():
        candidates.append(run_dir / path)
        candidates.extend(parent / path for parent in run_dir.parents)
    keys: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate.absolute()
        keys.add(os.path.normcase(os.path.abspath(str(resolved))))
        keys.add(os.path.normcase(resolved.name))
    return keys


def _source_video_and_duration(
    context: dict[str, Any],
    run_dir: Path,
    *,
    recorded_run_dir: Path | None = None,
) -> tuple[float, Path]:
    manifest: dict[str, Any] = {}
    manifest_path_value = context.get("manifest_path")
    if manifest_path_value:
        manifest_path = resolve_run_path(
            manifest_path_value,
            run_root=run_dir,
            recorded_run_dir=recorded_run_dir,
        )
        if manifest_path.exists():
            manifest = _load_json_object(manifest_path)

    try:
        source_video = resolve_input_video(
            run_dir,
            manifest=manifest,
            context=context,
            recorded_run_dir=recorded_run_dir,
        )
    except FileNotFoundError as exc:
        raise PostMergeError(str(exc)) from exc

    total_duration_value = manifest.get("total_duration") or context.get("total_duration")
    total_duration = float(total_duration_value) if total_duration_value is not None else get_duration(source_video)
    return total_duration, source_video


def _source_indices_by_result_index(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    total_duration: float,
    config: dict[str, Any],
    content_type: str,
) -> dict[int, list[int]]:
    type_config = config.get(content_type, {})
    padding_config = get_padding_config(config, content_type)
    merge_gap = float(
        padding_config.get("merge_gap_seconds")
        or type_config.get("merge_gap_seconds", 10.0)
    )
    max_duration_raw = padding_config.get("max_song_seconds") or type_config.get("max_duration")
    max_duration = float(max_duration_raw) if max_duration_raw is not None else None

    raw_matches: list[dict[str, Any]] = []
    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        for group_indices in _split_indices_by_time_gap(segments, valid_indices, merge_gap):
            raw_matches.append({
                "title": match.title,
                "content_type": match.content_type,
                "start": segments[min(group_indices)].start,
                "end": segments[max(group_indices)].end,
                "segment_start_idx": min(group_indices),
                "segment_end_idx": max(group_indices),
                "confidence": match.confidence,
                "transcript": " ".join(segments[i].text for i in group_indices),
                "tags": match.tags,
                "description": match.description,
                "artist": match.artist,
                "lyrics_snippet": match.lyrics_snippet,
            })

    merged = _merge_adjacent_matches(raw_matches, merge_gap, max_duration=max_duration)
    built_results = build_content_results(segments, matches, total_duration, config, content_type)
    built_indexes = {result.index for result in built_results}
    return {
        index: list(range(item["segment_start_idx"], item["segment_end_idx"] + 1))
        for index, item in enumerate(merged, start=1)
        if index in built_indexes
    }


def _merge_output_stem(paths: list[Path]) -> str:
    return safe_path_part(
        "__merge__".join(path.stem for path in paths),
        fallback="merged_song",
        max_length=180,
    )


def _merged_title_artist_many(results: list[ContentResult]) -> tuple[str, str]:
    titles: list[str] = []
    for result in results:
        if result.title and result.title not in titles:
            titles.append(result.title)
    title = titles[0] if len(titles) == 1 else "+".join(titles)

    artists = [result.artist for result in results if result.artist]
    if artists and all(artist == artists[0] for artist in artists):
        artist = artists[0]
    else:
        artist = artists[0] if artists else ""
    return title, artist


def _merged_title_artist(first: ContentResult, second: ContentResult) -> tuple[str, str]:
    return _merged_title_artist_many([first, second])


def _force_single_song_config(config: dict[str, Any], total_duration: float) -> dict[str, Any]:
    """Keep the synthetic post-merge match from being split by the normal merge gap."""
    from copy import deepcopy

    local_config = deepcopy(config)
    forced_gap = max(float(total_duration), 0.0) + 1.0
    local_config.setdefault("padding", {})["merge_gap_seconds"] = forced_gap
    local_config.setdefault("song", {}).setdefault("padding", {})["merge_gap_seconds"] = forced_gap
    return local_config


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise PostMergeError(f"Could not find an unused output path for: {path}")
