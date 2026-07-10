# Local Model Docker Image

The main branch carries the Track 1 local-model Docker source, deterministic solvers, tool logic, and the exported Docker image tar through Git LFS.

## Included In This Branch

- `Dockerfile`
- `entrypoint.sh`
- `agent.py`
- `tools.json`
- `run_local_with_fireworks.sh`
- `local.env.example`
- `dist/ARTIFACT_LOCATION.md`
- `dist/gemma3-1b-qat-track1.tar` tracked with Git LFS

The agent uses:

- local model: `gemma3:1b-it-qat`
- deterministic Track 1 solvers before model calls
- calculator/search-style tool handling inside `agent.py`
- optional Fireworks fallback through runtime environment variables

## Docker Image Artifact

The exported Docker image is committed through Git LFS because normal GitHub file storage rejects files above 100 MB.

```text
dist/gemma3-1b-qat-track1.tar
```

If cloning this branch, install Git LFS first so the tar downloads as the real binary instead of an LFS pointer.

```bash
git lfs install
git lfs pull
```

## Load Image

```bash
docker load -i dist/gemma3-1b-qat-track1.tar
```

## Run Loaded Image

```bash
mkdir -p input output
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  gemma3-1b-qat-track1:latest
```

The container reads:

```text
input/tasks.json
```

and writes:

```text
output/results.json
```

## Adaptive Local Engine

See `LOCAL_ENGINE_REPORT.md` for the certified local interface, tool design, and reproducible benchmark commands. The local engine is offline by default and leaves team routing/frontend code untouched.
