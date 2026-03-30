#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse


REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

ALLOWED_MODULES = {
    "llm_model",
    "video_backbone",
    "caption_gen",
    "window_gen",
    "feedback_caption",
    "global_local_emb_feedback",
}

REQUIRED_FIELDS = {
    "key",
    "repo_id",
    "repo_type",
    "module",
    "purpose",
    "source",
    "local_subdir",
    "license",
    "splits",
}

OPTIONAL_FIELDS = {
    "revision",
    "allow_patterns",
    "notes",
    "language",
    "modality",
}

ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def _add_error(errors: List[Dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_url(value: Any) -> bool:
    if not _is_non_empty_string(value):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_local_subdir(value: Any) -> Optional[str]:
    if not _is_non_empty_string(value):
        return "must be a non-empty string"
    if "\\" in value:
        return "must use forward slashes only"
    if re.match(r"^[A-Za-z]:", value):
        return "must be relative (Windows-style absolute path is not allowed)"

    posix = PurePosixPath(value)
    if posix.is_absolute():
        return "must be relative (absolute path is not allowed)"
    if not posix.parts:
        return "must not be empty"
    if any(part in {"", ".", ".."} for part in posix.parts):
        return "must not contain empty, '.' or '..' path segments"
    return None


def _validate_string_or_string_list(value: Any, field_name: str) -> Optional[str]:
    if isinstance(value, str):
        return None if value.strip() else f"'{field_name}' string must be non-empty"
    if isinstance(value, list):
        if not value:
            return f"'{field_name}' list must not be empty"
        for idx, item in enumerate(value):
            if not _is_non_empty_string(item):
                return f"'{field_name}[{idx}]' must be a non-empty string"
        return None
    return f"'{field_name}' must be a string or list of strings"


def validate_registry_payload(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "top-level JSON must be an object")
        return errors

    version = payload.get("version")
    if not isinstance(version, int):
        _add_error(errors, "version", "must be an integer")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        _add_error(errors, "datasets", "must be a list")
        return errors

    seen_keys: Dict[str, str] = {}
    seen_module_subdir: Dict[tuple[str, str], str] = {}

    for idx, item in enumerate(datasets):
        base = f"datasets[{idx}]"
        if not isinstance(item, dict):
            _add_error(errors, base, "must be an object")
            continue

        field_names = set(item.keys())
        missing = sorted(REQUIRED_FIELDS - field_names)
        for name in missing:
            _add_error(errors, f"{base}.{name}", "missing required field")

        unknown = sorted(field_names - ALLOWED_FIELDS)
        for name in unknown:
            _add_error(errors, f"{base}.{name}", "unknown field is not allowed")

        key = item.get("key")
        if "key" in item:
            if not _is_non_empty_string(key):
                _add_error(errors, f"{base}.key", "must be a non-empty string")
            elif not SNAKE_CASE_RE.match(key):
                _add_error(errors, f"{base}.key", "must be lowercase snake_case")
            else:
                prev = seen_keys.get(key)
                if prev is not None:
                    _add_error(errors, f"{base}.key", f"duplicate key '{key}' (first seen at {prev})")
                else:
                    seen_keys[key] = f"{base}.key"

        repo_id = item.get("repo_id")
        if "repo_id" in item:
            if not _is_non_empty_string(repo_id):
                _add_error(errors, f"{base}.repo_id", "must be a non-empty string")
            elif not REPO_ID_RE.match(repo_id):
                _add_error(errors, f"{base}.repo_id", "must match 'owner/name' format")

        repo_type = item.get("repo_type")
        if "repo_type" in item:
            if not _is_non_empty_string(repo_type):
                _add_error(errors, f"{base}.repo_type", "must be a non-empty string")
            elif repo_type not in {"dataset", "model"}:
                _add_error(errors, f"{base}.repo_type", "must be one of: dataset, model")

        module = item.get("module")
        module_valid = False
        if "module" in item:
            if not _is_non_empty_string(module):
                _add_error(errors, f"{base}.module", "must be a non-empty string")
            elif not SNAKE_CASE_RE.match(module):
                _add_error(errors, f"{base}.module", "must be lowercase snake_case")
            elif module not in ALLOWED_MODULES:
                _add_error(
                    errors,
                    f"{base}.module",
                    f"must be one of: {', '.join(sorted(ALLOWED_MODULES))}",
                )
            else:
                module_valid = True

        purpose = item.get("purpose")
        if "purpose" in item and not _is_non_empty_string(purpose):
            _add_error(errors, f"{base}.purpose", "must be a non-empty string")

        source = item.get("source")
        if "source" in item and not _validate_url(source):
            _add_error(errors, f"{base}.source", "must be a valid http/https URL")

        local_subdir = item.get("local_subdir")
        local_subdir_valid = False
        if "local_subdir" in item:
            local_subdir_error = _validate_local_subdir(local_subdir)
            if local_subdir_error is not None:
                _add_error(errors, f"{base}.local_subdir", local_subdir_error)
            else:
                local_subdir_valid = True

        license_name = item.get("license")
        if "license" in item and not _is_non_empty_string(license_name):
            _add_error(errors, f"{base}.license", "must be a non-empty string (use 'unknown' if needed)")

        splits = item.get("splits")
        if "splits" in item:
            if not isinstance(splits, list):
                _add_error(errors, f"{base}.splits", "must be a list")
            elif len(splits) == 0:
                _add_error(errors, f"{base}.splits", "must contain at least one split")
            else:
                for split_idx, split_name in enumerate(splits):
                    split_path = f"{base}.splits[{split_idx}]"
                    if not _is_non_empty_string(split_name):
                        _add_error(errors, split_path, "must be a non-empty string")
                    elif not SNAKE_CASE_RE.match(split_name):
                        _add_error(errors, split_path, "must be lowercase snake_case")

        if "revision" in item and not _is_non_empty_string(item["revision"]):
            _add_error(errors, f"{base}.revision", "must be a non-empty string")

        if "allow_patterns" in item:
            allow_patterns = item["allow_patterns"]
            if not isinstance(allow_patterns, list):
                _add_error(errors, f"{base}.allow_patterns", "must be a list of strings")
            else:
                for pat_idx, pattern in enumerate(allow_patterns):
                    if not _is_non_empty_string(pattern):
                        _add_error(errors, f"{base}.allow_patterns[{pat_idx}]", "must be a non-empty string")

        if "notes" in item and not _is_non_empty_string(item["notes"]):
            _add_error(errors, f"{base}.notes", "must be a non-empty string")

        if "language" in item:
            message = _validate_string_or_string_list(item["language"], "language")
            if message is not None:
                _add_error(errors, f"{base}.language", message)

        if "modality" in item:
            message = _validate_string_or_string_list(item["modality"], "modality")
            if message is not None:
                _add_error(errors, f"{base}.modality", message)

        if module_valid and local_subdir_valid:
            pair = (module, local_subdir)
            prev = seen_module_subdir.get(pair)
            if prev is not None:
                _add_error(
                    errors,
                    f"{base}.local_subdir",
                    f"duplicate (module, local_subdir)=({module}, {local_subdir}) (first seen at {prev})",
                )
            else:
                seen_module_subdir[pair] = f"{base}.local_subdir"

    return errors


def _write_json_report(path: Path, registry: Path, errors: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "valid": len(errors) == 0,
        "registry": registry.as_posix(),
        "error_count": len(errors),
        "errors": list(errors),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate datasets_registry.json against strict operational schema.")
    parser.add_argument(
        "--registry",
        type=str,
        default="datasets_registry.json",
        help="Path to dataset registry JSON file.",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default="",
        help="Optional path to write machine-readable JSON validation report.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    errors: List[Dict[str, str]]
    try:
        raw = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors = [{"path": "registry", "message": f"unable to read file: {exc}"}]
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors = [{"path": "registry", "message": f"invalid JSON: {exc}"}]
        else:
            errors = validate_registry_payload(payload)

    if args.json_report:
        _write_json_report(Path(args.json_report), registry_path, errors)

    if errors:
        for err in errors:
            print(f"ERROR {err['path']}: {err['message']}")
        print(f"Validation failed: {len(errors)} error(s).")
        return 1

    print("Validation passed: registry schema is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
