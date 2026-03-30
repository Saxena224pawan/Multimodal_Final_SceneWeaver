#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_compat import load_validate_models_registry


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _download_cmd(repo_id: str, local_path: Path, requires_auth: bool) -> str:
    cmd = f"huggingface-cli download {shlex.quote(repo_id)} --local-dir {shlex.quote(str(local_path))}"
    if requires_auth:
        return f"# requires accepted license/login\n{cmd}"
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link model registry entries to runtime paths for all pipeline modules."
    )
    parser.add_argument(
        "--registry",
        default="models_registry.json",
        help="Path to model registry JSON.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve local_subdir.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pipeline/model_links.json",
        help="Output linked model manifest path.",
    )
    parser.add_argument(
        "--strict-exists",
        action="store_true",
        help="Fail if any primary model local path does not exist.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional machine-readable execution report path.",
    )
    return parser.parse_args()


def _group_models(models: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in models:
        module = item["module"]
        slot = grouped.setdefault(module, {"primary": None, "fallbacks": [], "aux": []})
        role = item["role"]
        if role == "primary":
            slot["primary"] = item
        elif role == "fallback":
            slot["fallbacks"].append(item)
        else:
            slot["aux"].append(item)
    return grouped


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    report_path = Path(args.json_report) if args.json_report else None

    validator = load_validate_models_registry(ROOT)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report = {
            "ok": False,
            "errors": [{"path": "registry", "message": f"file not found: {registry_path}"}],
        }
        if report_path:
            _write_json(report_path, report)
        print(f"[ERROR] file not found: {registry_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        report = {
            "ok": False,
            "errors": [{"path": "registry", "message": f"invalid JSON: {exc}"}],
        }
        if report_path:
            _write_json(report_path, report)
        print(f"[ERROR] invalid JSON in {registry_path}: {exc}", file=sys.stderr)
        return 1

    errors = validator.validate_models_registry_payload(payload)
    if errors:
        report = {"ok": False, "errors": errors}
        if report_path:
            _write_json(report_path, report)
        print("[ERROR] models registry validation failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item['path']}: {item['message']}", file=sys.stderr)
        return 1

    grouped = _group_models(payload["models"])
    module_links: Dict[str, Dict[str, Any]] = {}
    missing_primary: List[str] = []
    download_commands: List[str] = []

    for module, slot in sorted(grouped.items()):
        primary = slot["primary"]
        primary_path = project_root / primary["local_subdir"]
        primary_exists = primary_path.exists()
        if not primary_exists:
            missing_primary.append(module)
            download_commands.append(
                _download_cmd(primary["repo_id"], primary_path, bool(primary["requires_auth"]))
            )

        fallback_entries: List[Dict[str, Any]] = []
        for fb in slot["fallbacks"]:
            fb_path = project_root / fb["local_subdir"]
            fb_exists = fb_path.exists()
            fallback_entries.append(
                {
                    "key": fb["key"],
                    "repo_id": fb["repo_id"],
                    "source": fb["source"],
                    "license": fb["license"],
                    "requires_auth": fb["requires_auth"],
                    "local_path": str(fb_path),
                    "exists": fb_exists,
                }
            )
            if not fb_exists:
                download_commands.append(
                    _download_cmd(fb["repo_id"], fb_path, bool(fb["requires_auth"]))
                )

        aux_entries: List[Dict[str, Any]] = []
        for aux in slot["aux"]:
            aux_path = project_root / aux["local_subdir"]
            aux_exists = aux_path.exists()
            aux_entries.append(
                {
                    "key": aux["key"],
                    "repo_id": aux["repo_id"],
                    "source": aux["source"],
                    "license": aux["license"],
                    "requires_auth": aux["requires_auth"],
                    "local_path": str(aux_path),
                    "exists": aux_exists,
                }
            )
            if not aux_exists:
                download_commands.append(
                    _download_cmd(aux["repo_id"], aux_path, bool(aux["requires_auth"]))
                )

        module_links[module] = {
            "selected": {
                "key": primary["key"],
                "repo_id": primary["repo_id"],
                "source": primary["source"],
                "license": primary["license"],
                "requires_auth": primary["requires_auth"],
                "local_path": str(primary_path),
                "exists": primary_exists,
            },
            "fallbacks": fallback_entries,
            "aux": aux_entries,
        }

    manifest = {
        "version": 1,
        "registry_version": payload["version"],
        "project_root": str(project_root),
        "modules": module_links,
        "missing_primary_modules": missing_primary,
        "download_commands": download_commands,
    }
    _write_json(output_path, manifest)

    if args.strict_exists and missing_primary:
        report = {
            "ok": False,
            "errors": [
                {
                    "path": "modules.primary",
                    "message": f"missing local model path for modules: {', '.join(missing_primary)}",
                }
            ],
            "output": str(output_path),
        }
        if report_path:
            _write_json(report_path, report)
        print(
            "[ERROR] missing local model path for primary modules: "
            + ", ".join(missing_primary),
            file=sys.stderr,
        )
        print(f"[INFO] linked manifest still written: {output_path}", file=sys.stderr)
        return 1

    report = {
        "ok": True,
        "errors": [],
        "output": str(output_path),
        "missing_primary_modules": missing_primary,
        "download_commands_count": len(download_commands),
    }
    if report_path:
        _write_json(report_path, report)

    print(
        f"[OK] model links generated: {output_path} "
        f"(missing_primary={len(missing_primary)}, download_cmds={len(download_commands)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
