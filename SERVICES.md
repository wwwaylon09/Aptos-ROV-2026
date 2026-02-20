# Split ROV Services

The old `bottom-side.py` monolith has been split into two independently restartable services:

- `scripts/run_control_server.py` → `rov/control_server.py` (TCP control on port `5000`, no CV code).
- `scripts/run_video_server.py` → `rov/video_server.py` (Flask camera server with `/`, `/status`, `/video/<camera>`, `/detections`).

Shared constants and environment parsing live in `rov/shared_config.py`.

## Start either service independently

```bash
./scripts/rov_services.sh control
./scripts/rov_services.sh video
```

## Start both (backward-compatible behavior)

```bash
./scripts/rov_services.sh both
# or
python3 bottom-side.py
```

## Health and failure isolation

- Control service emits periodic health logs (`packets`, `failsafe_events`, packet age) and can continue even if video service fails.
- Video service emits periodic camera health logs and keeps reconnecting cameras without affecting thruster control.
- Video `/status` includes service uptime, per-camera online/failure details, and latest non-stale detections by camera.
- Video `/detections` accepts CV packets via `POST` and exposes JSON schema + latest per-camera detections via `GET`.
