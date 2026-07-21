# Traffic Cone Detection Project

## Project background

This repository trains and runs an oriented-bounding-box detector for traffic
cones. `script/infer_obb.py` performs inference, groups visible cones into
routes (the displayed automatic ROIs), applies missing/shift/fallen rules, and
writes annotated video plus diagnostic events. The main regression fixture is
`script/test-video/DJI_1m22s_1m36s.mp4`, which contains forward drone motion and
therefore changing perspective and visible route membership.

## Output contracts

- Keep drawing a confirmed automatic ROI even when it currently contains only
  two cones. This is part of the customer-facing detection demonstration.
- A route with fewer than three currently detected cones must not produce any
  stable business alert or alert-log entry.
- Keep `events.jsonl` as the detailed per-frame diagnostic record, including raw
  detections and raw rule results. It is intended for programmatic review.
- Also write structured and human-review alert logs. Human-facing entries use a
  sequential alert number, video time in seconds, `ROI_<stable id>`, and a plain
  alert type such as `MISSING`, `SHIFT`, or `FALLEN`; do not include centers,
  polygons, or other geometry.

## Confirmed bugs

The DJI fixture previously produced two `MISSING` alerts near 0.934 and 1.001
seconds. The physical route had not changed at those exact frames: two gaps
oscillated around the 1.8 ratio threshold as perspective changed. The old code
tracked only a route-level nominal gap, so a one-frame threshold crossing was
immediately published and the same gap had no persistent identity across
frames.

Short or edge routes are a separate case. A one-frame tentative route was
correctly excluded from stable alerts, but a visible two-cone edge route could
be confirmed and displayed. Displaying it is required; recording business
alerts for it is not.

The old event timestamp used wall-clock time. That is not useful for manually
locating an alert in a video; review logs must use media time derived from frame
index and FPS.

## Required solution

- Assign a stable ID to each gap inside a stable route by tracking its endpoint
  cone identities across adjacent frames.
- When a cone enters or leaves a route, suppress the affected adjacent gaps for
  five frames (configurable within the requested three-to-five-frame range).
- Confirm a candidate alert only after it is positive in at least three of the
  latest five observations for the same stable gap.
- Once confirmed, keep the alert active until the same gap has been normal for
  three consecutive observations.
- Apply the minimum-three-cone business gate after ROI tracking, without
  changing route grouping or ROI drawing.
- Record only alert activation transitions in the concise alert logs so a
  persistent condition does not create one human entry per video frame.

## Oblique drone distance correction

Inference defaults to a flat-ground pinhole projection for a roll-stabilized
drone camera. The default optical-axis depression is 45 degrees and the default
horizontal field of view is 84 degrees. Missing-gap ratios and displacement
rules use projected ground coordinates; route grouping, ROI drawing, and route
tracking remain in image pixels.

Pitch alone cannot recover meters. Without camera height, corrected distances
are expressed in `camera_heights` and are valid for relative gap ratios. With
`--drone-altitude-m`, corrected distances are meters. Camera pitch, field of
view, height, focal length, and distance unit must be recorded in every event.
Keep original pixel distances alongside corrected values for diagnosis. The
model assumes a level ground plane and negligible camera roll; deployments
outside those assumptions require a full calibrated homography.

## Verification

Maintain focused unit tests for gap identity, membership cooldown, voting,
clearing, minimum route size, and alert-log formatting. After unit tests pass,
re-run the DJI fixture and inspect both `events.jsonl` and the concise alert logs
around 0-2 seconds. Raw events may still contain threshold candidates; they
must not appear as stable or human-review alerts unless the temporal policy and
minimum route size are satisfied.

## Current artifacts

- Conda environment: `yolo-obb` from `script/environment.yml`.
- Fused four-class dataset: `dataset/2026-07-15/data.yaml` with `cone`,
  `traffic_sign`, `person`, and `vest`.
- Recommended weights:
  `script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt`.
- Resume checkpoint:
  `script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/last.pt`.
- Main drone regression video: `script/test-video/DJI_1m22s_1m36s.mp4`.
- Latest verified 45-degree output:
  `script/runs/infer_videos_perspective45_final/DJI_1m22s_1m36s/`.

## Current commands

Run commands from the repository root.

Create the environment once:

```bash
conda env create -f script/environment.yml
```

Validate the existing fused dataset:

```bash
conda run -n yolo-obb python script/validate_dataset.py \
  --data-root dataset/2026-07-15
```

Fresh training with the current production configuration:

```bash
conda run -n yolo-obb python script/train_obb.py \
  --data dataset/2026-07-15/data.yaml \
  --model yolov8s-obb.pt \
  --device 0 \
  --epochs 150 \
  --batch 16 \
  --imgsz 960 \
  --workers 8 \
  --patience 30 \
  --cache \
  --name traffic_cone_person_vest_yolov8s_obb_960_v2
```

After this fresh run completes, its candidate weight is
`script/runs/traffic_cone_person_vest_yolov8s_obb_960_v2/weights/best.pt`.
Do not replace the currently recommended weight until the new run has been
validated on the test split and the DJI regression video.

Resume an interrupted training run:

```bash
conda run -n yolo-obb python script/train_obb.py \
  --resume script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/last.pt \
  --device 0
```

Drone inference with the default relative 45-degree correction. Distances are
in `camera_heights` because no measured altitude is supplied:

```bash
conda run -n yolo-obb python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 960 \
  --device 0 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --output script/runs/infer_videos_perspective45/DJI_1m22s_1m36s
```

Metric drone inference requires measured camera height above the cone ground
plane. Replace `<measured-height-m>` and, where known, replace 84 with the
camera's effective horizontal field of view after any crop:

```bash
conda run -n yolo-obb python script/infer_obb.py \
  --weights script/runs/traffic_cone_person_vest_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 960 \
  --device 0 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --drone-altitude-m <measured-height-m> \
  --output script/runs/infer_videos_perspective45_metric/DJI_1m22s_1m36s
```

The output directory contains the annotated video, detailed `events.jsonl`,
structured activation-only `alert.jsonl`, and human-review `alert.log`.
