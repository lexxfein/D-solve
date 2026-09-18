import type { Treatment } from "./types";
export type { Treatment } from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function localPreview(file: File, treatment: Treatment, intensity: number): Promise<string> {
  const source = URL.createObjectURL(file);
  const img = new Image();
  await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = reject; img.src = source; });
  const canvas = document.createElement("canvas"); canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("Browser preview is not supported.");
  ctx.filter = treatment === "whitening" ? `brightness(${1 + intensity / 250}) saturate(${1 - intensity / 400})` : `brightness(${1 + intensity / 300}) contrast(${1 + intensity / 500})`;
  ctx.drawImage(img, 0, 0);
  if (treatment === "combined") { ctx.filter = `brightness(${1 + intensity / 350}) saturate(${1 - intensity / 500})`; ctx.globalAlpha = 0.22; ctx.drawImage(img, 0, 0); }
  URL.revokeObjectURL(source);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export async function simulateSmile(file: File, treatment: Treatment, intensity: number): Promise<{ url: string; demo: boolean }> {
  const form = new FormData(); form.append("file", file); form.append("treatment", treatment); form.append("intensity", String(intensity));
  try {
    const controller = new AbortController(); const timer = window.setTimeout(() => controller.abort(), 12000);
    const response = await fetch(`${API_BASE_URL}/api/simulate`, { method: "POST", body: form, signal: controller.signal }); window.clearTimeout(timer);
    if (!response.ok) throw new Error(await response.text());
    return { url: URL.createObjectURL(await response.blob()), demo: false };
  } catch {
    return { url: await localPreview(file, treatment, intensity), demo: true };
  }
}

async function postModel(path: string, form: FormData): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = "3D alignment request failed.";
    try { const body = await response.json(); detail = body.detail ?? detail; } catch { /* ignore non-JSON errors */ }
    throw new Error(detail);
  }
  return response.blob();
}

export interface AutoAlignmentResult {
  blob: Blob;
  transform: { strength: number; translation: number[]; rotation: number[]; scale: number; deformation?: string; note: string };
}

export async function getModelInfo(file: File) {
  const form = new FormData(); form.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/alignment/info`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = "Could not analyze the 3D model.";
    try { const body = await response.json(); detail = body.detail ?? detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return response.json() as Promise<{ vertices: number; faces: number; meshes: number; bounds_min: number[]; bounds_max: number[]; center: number[]; size: number[] }>;
}

export async function autoAlignModel(file: File, strength: number): Promise<AutoAlignmentResult> {
  const form = new FormData(); form.append("file", file); form.append("strength", String(strength));
  const response = await fetch(`${API_BASE_URL}/api/alignment/auto`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = "Automatic 3D alignment failed.";
    try { const body = await response.json(); detail = body.detail ?? detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  const blob = await response.blob();
  let transform: AutoAlignmentResult["transform"] = { strength, translation: [0, 0, 0], rotation: [0, 0, 0], scale: 1, note: "" };
  const header = response.headers.get("X-Alignment-Transform");
  if (header) {
    try { transform = JSON.parse(header) as AutoAlignmentResult["transform"]; } catch { /* ignore malformed optional header */ }
  }
  return { blob, transform };
}

export async function transformModel(file: File, values: { tx: number; ty: number; tz: number; rx: number; ry: number; rz: number; scale: number }) {
  const form = new FormData(); form.append("file", file);
  Object.entries(values).forEach(([key, value]) => form.append(key, String(value)));
  return postModel("/api/alignment/transform", form);
}
