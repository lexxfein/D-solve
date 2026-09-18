# Dentaform — Digital Smile Design Frontend

This frontend is a complete browser-side workflow for the Digital Smile Design project.

## Included screens
- Dashboard with live local case/patient counts
- New Smile Design workflow
- Patient management: add, edit, search, delete
- Smile Designs: search, view before/after, delete
- Reports: treatment summary + CSV export
- Settings: clinic name, notification preference, local-data reset
- Responsive sidebar/mobile navigation

## Data storage
Until the real backend/database is available, patient and design records are stored in browser `localStorage`. This makes the frontend usable for demos without a backend.

## Computer vision connection
`src/api.ts` tries the FastAPI endpoint at `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`). If the backend is unavailable, the frontend shows an explicitly labelled demo preview so the rest of the UI remains usable. The CV files are not modified by this frontend integration.

## Run
```bash
npm install
npm run dev
```

Optional `.env`:
```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```
