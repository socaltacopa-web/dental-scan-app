# Dental Scan — GitHub + Railway Prototype

This project is packaged so you can put the files directly into a GitHub repository and deploy that repository to Railway.

## What is included

- phone-friendly guided dental capture web app
- FastAPI backend
- scan upload endpoint
- server-side OpenCV reconstruction
- Open3D global alignment
- optional Poisson mesh generation
- processing status endpoint
- artifact downloads
- browser point-cloud viewer
- Railway `Dockerfile`
- Railway health-check config


## Front and back camera support

The capture page now lets the user switch between:

- **Back camera** — recommended because it is usually sharper and has better close-focus performance.
- **Front/selfie camera** — useful when scanning yourself because you can see the screen while positioning the phone.

The front-camera preview is mirrored for easier positioning, but captured image data is kept in the camera's normal orientation for reconstruction.


## Important

This is **research/prototype software**. Do not use it for diagnosis, treatment planning, measurements, crowns, aligners, dentures, surgical guides, or fabrication.

Do not upload real patient health information until you have implemented the privacy, access-control, audit, retention, encryption, and regulatory controls required for your intended use.

## GitHub

Create a new repository, then upload the **contents of this folder** to the repository root.

The repository root should look like:

```text
app/
pipeline/
static/
Dockerfile
railway.toml
requirements.txt
README.md
```

## Railway deployment

1. Create a new Railway project.
2. Choose **Deploy from GitHub repo**.
3. Select this repository.
4. Railway should detect the root `Dockerfile`.
5. Generate a public domain for the service.
6. Add a persistent Railway Volume and mount it to:

```text
/data
```

The backend automatically uses `RAILWAY_VOLUME_MOUNT_PATH` when Railway provides it.

Without a persistent volume, uploaded scans and reconstructed models should be treated as temporary.

## Environment variables

These are optional:

```text
AUTO_PROCESS=1
ENABLE_MESH=1
MAX_UPLOAD_MB=100
PROCESS_TIMEOUT_SECONDS=1800
```

If a Railway volume is attached, the app automatically reads its mount path. You can also manually define:

```text
DATA_DIR=/data
```

## Local development

Create a virtual environment, then:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

## API

### Health

```text
GET /health
```

### Submit a scan

```text
POST /api/scans
```

Multipart fields:

```text
file       .dscan.json file
case_name  optional label
```

### Scan status

```text
GET /api/scans/{scan_id}
```

### Re-run processing

```text
POST /api/scans/{scan_id}/process
```

### Download artifact

```text
GET /api/scans/{scan_id}/artifacts/{path}
```

## Processing pipeline

After submission:

```text
phone capture
    ↓
input.dscan.json
    ↓
reconstruct_sweeps.py
    ↓
per-sweep dense point clouds
    ↓
align_global.py
    ↓
merged_mouth.ply
    ↓
optional merged_mouth_mesh.ply
```

## Storage layout

```text
data/
  scans/
    <scan-id>/
      input.dscan.json
      status.json
      processing.log
      reconstruction/
      global/
```

## What should be built next before real patient use

1. patient/dentist authentication
2. separate dentist dashboard
3. authorization so dentists only see assigned cases
4. encrypted/object storage strategy
5. database instead of JSON status files
6. upload retention/deletion controls
7. audit logging
8. clinical validation against professional intraoral scans
9. a trained tooth/gum segmentation model
10. better depth estimation and bundle adjustment

For a prototype, keeping everything in one Railway service plus one persistent volume is the simplest deployment. For production, the processor should eventually move to a separate worker service or job queue so heavy 3D processing cannot block the main web application.
