# Vault Storage Migration

## Overview
Large model files have been moved from the workspace to HPC vault storage to free up workspace disk space.

## Moved Directories
- `models/` (63GB) → `/home/vault/v123be/v123be37/Multimodal_Final_SceneWeaver/models/`
- `LLM_MODEL/` (12GB) → `/home/vault/v123be/v123be37/Multimodal_Final_SceneWeaver/LLM_MODEL/`

## Implementation
- Original directories moved to vault storage
- Symlinks created to maintain original paths
- All code continues to work without modification
- Models remain accessible through symlinks

## Benefits
- Workspace size reduced from ~75GB to ~2.3GB
- Large files stored on dedicated vault storage with 858TB available
- No impact on functionality or performance
- Version control unaffected (models/ already in .gitignore)

## Verification
- Symlinks confirmed working
- Python imports successful
- No configuration changes required