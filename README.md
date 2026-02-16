# Kepler ROV

A robotics team based out of Aptos High School. We build an underwater ROV to compete in the MATE ROV competition

## Metashape from known-good image folders

Use `metashape_from_image_folder.py` when you want to validate the Metashape pipeline using a folder of images that already reconstructs well.

Example:

```bash
python metashape_from_image_folder.py \
  --images-dir /path/to/known_good_images \
  --output-dir ./metashape_runs/known_good_test \
  --image-glob "*.jpg"
```

If Metashape is not on your PATH, pass `--metashape-executable` with the full path to `metashape.exe`.
