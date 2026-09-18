# Digital Smile Design — Integrated Project

This folder combines the **Dentaform TypeScript/React frontend** with the team's
**computer-vision modules** and a small FastAPI bridge.

## Architecture

Browser (React + TypeScript)
        |
        | POST /api/simulate (multipart/form-data)
        v
FastAPI backend
        |
        +--> cv/whitening/whitening.py
        +--> cv/alignment/alignment.py
        +--> cv/veneer/veneer.py
        |
        v
JPEG result -> browser

## Project structure

- `frontend/` — Dentaform React + TypeScript + Vite UI
- `backend/main.py` — API adapter between frontend and CV code
- `backend/requirements.txt` — Python dependencies
- `cv/` — original team's MediaPipe/OpenCV processing code
- `frontend/src/api.ts` — frontend API client
- `frontend/src/pages/NewDesignPage.tsx` — upload, treatment selection, intensity and before/after UI

## Run it

### 1. Backend

From the project root:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API should be available at `http://127.0.0.1:8000`.

Health check:

```text
GET http://127.0.0.1:8000/api/health
```

### 2. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`.

The frontend already defaults to the backend URL above. If needed, copy
`frontend/.env.example` to `frontend/.env` and change `VITE_API_BASE_URL`.

## Treatment API

`POST /api/simulate`

Form fields:

- `file`: image file
- `treatment`: `whitening`, `alignment`, `veneers`, or `combined`
- `intensity`: integer from 0–100

The response is a JPEG image.

## Compatibility status

Before integration, the two supplied projects were **not directly compatible**:

1. The Dentaform frontend only created a local browser preview and explicitly had no backend connection.
2. The supplied `D-Solve/backend/` Python adapter files were empty.
3. The supplied `D-Solve/frontend/` files were empty and therefore were not a usable bridge.
4. The CV code exposed Python functions (`whiten_smile`, `align_smile`, `veneer_smile`) but the TypeScript frontend cannot call Python functions directly from the browser.

The integration fixes this by adding the FastAPI HTTP boundary and wiring the existing
TypeScript UI to it.

## Important note

The CV code supplied by the team is preserved rather than rewritten. The backend is
an adapter layer around it. The visual output should be treated as a simulation for
communication, not as a clinical prediction.
