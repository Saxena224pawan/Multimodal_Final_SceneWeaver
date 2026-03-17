#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
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

ALLOWED_ROLES = {"primary", "fallback", "aux"}

REQUIRED_FIELDS = {
    "key",
    "repo_id",
    "repo_type",
    "module",
    "role",
    "purpose",
    "source",
    "local_subdir",
    "license",
    "requires_auth",
}

OPTIONAL_FIELDS = {
    "revision",
    "allow_patterns",
    "notes",
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
    posix = PurePosixPath(value)
    if posix.is_absolute():
        return "must be relative"
    if any(part in {"", ".", ".."} for part in posix.parts):
        return "must not contain empty, '.', or '..' segments"
    return None


def validate_models_registry_payload(payload: Any) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        _add_error(errors, "$", "top-level JSON must be an object")
        return errors

    version = payload.get("version")
    if not isinstance(version, int):
        _add_error(errors, "version", "must be an integer")

    models = payload.get("models")
    if not isinstance(models, list):
        _add_error(errors, "models", "must be a list")
        return errors

    seen_keys: Dict[str, str] = {}
    module_primary_count: Dict[str, int] = {}
    seen_module_local_subdir: Dict[tuple[str, str], str] = {}

    for idx, item in enumerate(models):
        base = f"models[{idx}]"
        if not isinstance(item, dict):
            _add_error(errors, base, "must be an object")
            continue

        fields = set(item.keys())
        for missing in sorted(REQUIRED_FIELDS - fields):
            _add_error(errors, f"{base}.{missing}", "missing required field")
        for unknown in sorted(fields - ALLOWED_FIELDS):
            _add_error(errors, f"{base}.{unknown}", "unknown field is not allowed")

        key = item.get("key")
        if "key" in item:
            if not _is_non_empty_string(key):
                _add_error(errors, f"{base}.key", "must be a non-empty string")
            elif not SNAKE_CASE_RE.match(key):
                _add_error(errors, f"{base}.key", "must be lowercase snake_case")
            else:
                prev = seen_keys.get(key)
                if prev:
                    _add_error(errors, f"{base}.key", f"duplicate key '{key}' (first seen at {prev})")
                else:
                    seen_keys[key] = f"{base}.key"

        repo_id = item.get("repo_id")
        if "repo_id" in item:
            if not _is_non_empty_string(repo_id):
                _add_error(errors, f"{base}.repo_id", "must be a non-empty string")
            elif not REPO_ID_RE.match(repo_id):
                _add_error(errors, f"{base}.repo_id", "must match owner/name")

        repo_type = item.get("repo_type")
        if "repo_type" in item:
            if repo_type != "model":
                _add_error(errors, f"{base}.repo_type", "must be 'model'")

        module = item.get("module")
        module_valid = False
        if "module" in item:
            if not _is_non_empty_string(module):
                _add_error(errors, f"{base}.module", "must be a non-empty string")
            elif module not in ALLOWED_MODULES:
                _add_error(
                    errors,
                    f"{base}.module",
                    f"must be one of: {', '.join(sorted(ALLOWED_MODULES))}",
                )
            else:
                module_valid = True

        role = item.get("role")
        if "role" in item:
            if role not in ALLOWED_ROLES:
                _add_error(errors, f"{base}.role", "must be one of: primary, fallback, aux")
            elif role == "primary" and module_valid:
                module_primary_count[module] = module_primary_count.get(module, 0) + 1

        if "purpose" in item and not _is_non_empty_string(item.get("purpose")):
            _add_error(errors, f"{base}.purpose", "must be a non-empty string")

        if "source" in item and not _validate_url(item.get("source")):
            _add_error(errors, f"{base}.source", "must be a valid http/https URL")

        local_subdir = item.get("local_subdir")
        local_subdir_valid = False
        if "local_subdir" in item:
            msg = _validate_local_subdir(local_subdir)
            if msg:
                _add_error(errors, f"{base}.local_subdir", msg)
            else:
                local_subdir_valid = True

        if "license" in item and not _is_non_empty_string(item.get("license")):
            _add_error(errors, f"{base}.license", "must be a non-empty string")

        if "requires_auth" in item and not isinstance(item.get("requires_auth"), bool):
            _add_error(errors, f"{base}.requires_auth", "must be a boolean")

        if "revision" in item and not _is_non_empty_string(item.get("revision")):
            _add_error(errors, f"{base}.revision", "must be a non-empty string")

        if "allow_patterns" in item:
            patterns = item.get("allow_patterns")
            if not isinstance(patterns, list):
                _add_error(errors, f"{base}.allow_patterns", "must be a list of strings")
            else:
                for p_idx, pattern in enumerate(patterns):
                    if not _is_non_empty_string(pattern):
                        _add_error(errors, f"{base}.allow_patterns[{p_idx}]", "must be a non-empty string")

        if "notes" in item and not _is_non_empty_string(item.get("notes")):
            _add_error(errors, f"{base}.notes", "must be a non-empty string")

        if module_valid and local_subdir_valid:
            k = (module, local_subdir)
            prev = seen_module_local_subdir.get(k)
            if prev:
                _add_error(
                    errors,
                    f"{base}.local_subdir",
                    f"duplicate (module, local_subdir) pair for {module}:{local_subdir} (first seen at {prev})",
                )
            else:
                seen_module_local_subdir[k] = f"{base}.local_subdir"

    for module in sorted(ALLOWED_MODULES):
        count = module_primary_count.get(module, 0)
        if count == 0:
            _add_error(errors, f"models.primary[{module}]", "missing primary model for module")
        elif count > 1:
            _add_error(errors, f"models.primary[{module}]", "multiple primary models found for module")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate models_registry.json for strict pipeline model linking."
    )
    parser.add_argument(
        "--registry",
        default="models_registry.json",
        help="Path to model registry JSON file.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional output path for machine-readable report.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    report_path = Path(args.json_report) if args.json_report else None

    if not registry_path.exists():
        payload = {
            "ok": False,
            "errors": [{"path": "registry", "message": f"file not found: {registry_path}"}],
        }
        if report_path:
            _write_json(report_path, payload)
        print(f"[ERROR] file not found: {registry_path}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report = {
            "ok": False,
            "errors": [{"path": "registry", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, report)
        print(f"[ERROR] invalid JSON in {registry_path}: {exc}", file=sys.stderr)
        return 1

    errors = validate_models_registry_payload(payload)
    if errors:
        report = {"ok": False, "errors": errors}
        if report_path:
            _write_json(report_path, report)
        print("[ERROR] models registry validation failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item['path']}: {item['message']}", file=sys.stderr)
        return 1

    report = {"ok": True, "errors": []}
    if report_path:
        _write_json(report_path, report)
    print(f"[OK] valid models registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
