# Dentaform frontend

## Start

```powershell
npm install
npm run dev
```

If this folder came from an older installation, clean the dependencies first:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue
npm install
```

The frontend uses React 19.2.8 and Three.js for the 3D model viewer.

Set `VITE_API_BASE_URL` in `.env` if the backend is not running at `http://127.0.0.1:8000`.
