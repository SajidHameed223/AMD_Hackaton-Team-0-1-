# Final Shareable Artifact

The GitHub branch `main` contains the final Track 1 local-model Docker image export as a Git LFS file:

```text
dist/gemma3-1b-qat-track1.tar
```

It also contains the Docker source, deterministic router/agent code, tools config, and run scripts.

Final image details:

```text
model: gemma3:1b-it-qat
image tag after load: gemma3-1b-qat-track1:latest
platform: linux/amd64
size: about 1.2 GB
storage: Git LFS
```

## Clone With LFS

```bash
git lfs install
git clone --branch main https://github.com/SajidHameed223/AMD_Hackaton-Team-0-1-.git
git lfs pull
```

## Load

```bash
docker load -i dist/gemma3-1b-qat-track1.tar
```

## Run

```bash
mkdir -p input output
```

Put tasks at:

```text
input/tasks.json
```

Run:

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  gemma3-1b-qat-track1:latest
```

The container writes:

```text
output/results.json
```

## Notes

- The image includes deterministic Track 1 solvers and tool handling in `agent.py`.
- Fireworks fallback is runtime-env based; no API key is committed.
- TurboQuant files are experimental and are not the final deliverable.
