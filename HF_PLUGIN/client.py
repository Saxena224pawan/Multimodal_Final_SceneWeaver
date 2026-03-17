from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class HFPluginConfig:
    cache_dir: Optional[str] = None
    token: Optional[str] = None


class HFPlugin:
    """Small wrapper around huggingface_hub for project use-cases."""

    def __init__(self, config: Optional[HFPluginConfig] = None):
        self.config = config or HFPluginConfig()

    def _load_hf_api(self):
        try:
            from huggingface_hub import login, snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            ) from exc
        return login, snapshot_download

    def login(self, token: Optional[str] = None) -> None:
        login, _ = self._load_hf_api()
        resolved = token or self.config.token or os.getenv("HF_TOKEN")
        if not resolved:
            raise ValueError("No HF token provided. Set HF_TOKEN or pass token explicitly.")
        login(token=resolved, add_to_git_credential=False)

    def download_model(
        self,
        repo_id: str,
        *,
        local_dir: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[Sequence[str]] = None,
    ) -> str:
        return self._download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            revision=revision,
            allow_patterns=allow_patterns,
        )

    def download_dataset(
        self,
        repo_id: str,
        *,
        local_dir: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[Sequence[str]] = None,
    ) -> str:
        return self._download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            revision=revision,
            allow_patterns=allow_patterns,
        )

    def _download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        local_dir: str,
        revision: Optional[str],
        allow_patterns: Optional[Sequence[str]],
    ) -> str:
        _, snapshot_download = self._load_hf_api()
        cache_dir = self.config.cache_dir or os.getenv("HF_HOME")
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            revision=revision,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
