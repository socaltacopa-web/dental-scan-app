# Dental Scan V4 — Global Alignment Prototype

V4 is the first version in this project that is designed to combine separate mouth sweeps into one 3D coordinate system.

## What changed

### Capture now includes alignment bridges
V4 uses 10 sweeps rather than 7.

The important new sweeps are:

- **Bite registration** — captures upper and lower teeth together as an anchor.
- **Upper bridge** — deliberately connects upper outer/front surfaces to upper biting and inside surfaces.
- **Lower bridge** — does the same for the lower arch.

The app repeatedly tells the person to preserve overlap. Global registration only works when two sweeps contain enough of the **same visible tooth geometry**.

### Sweep reconstruction is now internally consistent
`reconstruct_sweeps.py` chains the estimated camera poses.

V3 reconstructed adjacent frame pairs but did not put all pairs into a properly chained sweep-local coordinate system. V4 fixes that part.

### Global alignment
`align_global.py` uses Open3D to:

1. normalize each sweep cloud,
2. calculate FPFH geometric features,
3. perform coarse RANSAC registration,
4. refine with point-to-plane ICP,
5. construct a pose graph,
6. optimize the pose graph,
7. merge accepted sweep clouds,
8. remove statistical outliers,
9. optionally create a Poisson surface mesh.

Outputs include:

```text
global_v4/
  merged_mouth.ply
  merged_mouth_mesh.ply       # when --mesh is used and enough points exist
  alignment_report.json
```

## Run the capture app

```bash
cd dental_scan_v4
python serve_local.py
```

Open:

```text
http://localhost:8000
```

For a real phone, host the app over HTTPS.

## Install reconstruction dependencies

```bash
pip install -r requirements.txt
```

## Step 1 — reconstruct each sweep

After the app exports its `.dscan.json` file:

```bash
python reconstruct_sweeps.py dental-scan-v4-XXXXXXXX.dscan.json
```

This creates:

```text
reconstruction_v4/
```

with one dense point cloud per completed sweep.

## Step 2 — globally align the sweeps

```bash
python align_global.py reconstruction_v4 --package dental-scan-v4-XXXXXXXX.dscan.json --mesh
```

This creates:

```text
global_v4/
```

The main files are:

```text
merged_mouth.ply
merged_mouth_mesh.ply
alignment_report.json
```

Open `viewer.html` and select `merged_mouth.ply`.

## What success looks like

The front teeth captured in several scans should land in roughly the same place after alignment. Upper bridge should connect the upper outer/front cloud to the upper biting/inside clouds. Lower bridge should do the same for the lower teeth. Bite registration should provide a connection between upper and lower.

## What can still fail

This remains a hard computer-vision problem.

Phone-only dental scans have several problems:

- enamel is shiny and produces moving highlights,
- individual teeth can have repetitive geometry,
- lips/tongue/cheeks occlude surfaces,
- monocular reconstruction has unknown scale,
- essential-matrix translations have scale ambiguity,
- feature registration can lock onto the wrong tooth,
- current segmentation is heuristic rather than a trained dental model,
- camera intrinsics are only approximated,
- no full bundle adjustment is included yet.

For that reason, `alignment_report.json` records the fitness of each pairwise registration rather than silently assuming everything aligned correctly.

## The next major technical step

The next version should focus less on adding capture screens and more on **accuracy**:

1. train/use a real tooth-and-gum segmentation model,
2. use a learned depth model fine-tuned on intraoral imagery,
3. globally optimize camera poses with bundle adjustment,
4. estimate per-surface uncertainty,
5. compare phone reconstructions against professional intraoral scans.

That validation comparison is what will tell us whether this can become clinically useful or remains a remote-visualization tool.

## Important

This is research software, not a medical device. It must not be used as a substitute for a clinically validated intraoral scanner or for diagnosis, treatment planning, measurements, or fabrication of dental devices.
