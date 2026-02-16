# Kepler ROV

A robotics team based out of Aptos High School. We build an underwater ROV to compete in the MATE ROV competition

## Camera 4 photogrammetry automation (Metashape)

Use `camera4_metashape_capture.py` on the laptop to run a start/stop capture workflow and then launch an automated Agisoft Metashape model build.

```bash
python camera4_metashape_capture.py \
  --stream-url "http://192.168.42.42:5001/video/Camera%204" \
  --output-root ./photogrammetry_sessions \
  --metashape-executable "C:/Program Files/Agisoft/Metashape Pro/metashape.exe"
```

Interactive commands in terminal:
- `start` begins saving frames.
- `stop` ends capture and starts Metashape processing.

The script exposes many tuning options for capture and Metashape stages (alignment, camera optimization, depth map filtering, mesh/texture, export format/CRS). See all options:

```bash
python camera4_metashape_capture.py --help
```
