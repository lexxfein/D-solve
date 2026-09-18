import type { Patient, SmileDesign } from "./types";

const PATIENTS_KEY = "dentaform_patients_v1";
const DESIGNS_KEY = "dentaform_designs_v1";

const starterPatients: Patient[] = [
  { id: "p1", name: "Ananya Menon", age: 28, email: "ananya@example.com", phone: "+91 98765 43210", notes: "Interested in whitening and veneer consultation.", createdAt: new Date().toISOString() },
  { id: "p2", name: "Rahul Nair", age: 34, email: "rahul@example.com", phone: "+91 91234 56789", notes: "Follow-up alignment case.", createdAt: new Date().toISOString() },
  { id: "p3", name: "Meera Joseph", age: 25, email: "meera@example.com", phone: "+91 99887 77665", notes: "Initial smile design consultation.", createdAt: new Date().toISOString() },
];

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) as T : fallback;
  } catch { return fallback; }
}

export function getPatients(): Patient[] { return read(PATIENTS_KEY, starterPatients); }
export function savePatients(items: Patient[]) { localStorage.setItem(PATIENTS_KEY, JSON.stringify(items)); }
export function getDesigns(): SmileDesign[] { return read(DESIGNS_KEY, []); }
export function saveDesigns(items: SmileDesign[]) { localStorage.setItem(DESIGNS_KEY, JSON.stringify(items)); }

export function makeId(prefix: string) { return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }
