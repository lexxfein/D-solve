# D-Solve — Digital Smile Design

D-Solve contains the original 2D smile-design CV workflow plus a separate 3D dental-model alignment workflow.

## 2D workflow

- Whitening remains on the existing OpenCV/MediaPipe pipeline.
- Veneers remains on the existing veneer pipeline.
- The 2D Alignment option has been removed from the treatment UI so alignment is handled only in the 3D workflow.

## 3D alignment

The **3D Alignment** tab accepts `.glb`, `.gltf`, `.obj`, and `.stl` models. The backend:

1. Reads the uploaded mesh/scene with `trimesh`.
2. Reports mesh, vertex and face counts.
3. Applies a smooth dental-arch normalization to the fused scan.
4. Allows manual rigid translation, rotation and scale.
5. Exports the transformed result as a GLB.

The automatic operation now applies a **visible, conservative dental-arch normalization** to the fused scan. It smoothly adjusts lateral arch spacing, upper/lower arch height, and tooth-band depth instead of relying on PCA alone (which can be nearly a no-op for an already-upright scan). It does not independently reposition segmented teeth and is intended for visual design alignment, not clinical prediction. Tooth-by-tooth movement requires separate tooth objects or a dedicated tooth-segmentation/modeling stage.

A copy of the supplied test model is included at `sample_models/teeth 3d model.glb`.

## Run locally

### Backend

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

Open a second PowerShell in `frontend`:

```powershell
npm install
npm run dev
```

If an older copy of the frontend has already created `node_modules` or a lockfile, use a clean install:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue
npm install
npm run dev
```

React is pinned to `19.2.8` because the selected Three.js integration requires a React 19 version below 19.3.

## API

- `GET /api/health`
- `POST /api/simulate` — 2D whitening/veneers/combined
- `POST /api/alignment/info` — inspect a 3D model
- `POST /api/alignment/auto` — automatic geometric alignment
- `POST /api/alignment/transform` — manual rigid transform + scale
