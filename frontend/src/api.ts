import type { Treatment } from "./types";
export type { Treatment } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function localPreview(file: File, treatment: Treatment, intensity: number): Promise<string> {
  const source = URL.createObjectURL(file);
  const img = new Image();
  await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = reject; img.src = source; });
  const canvas = document.createElement("canvas"); canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("Browser preview is not supported.");
  ctx.filter = treatment === "whitening" ? `brightness(${1 + intensity / 250}) saturate(${1 - intensity / 400})` : treatment === "veneers" ? `brightness(${1 + intensity / 300}) contrast(${1 + intensity / 500})` : `contrast(${1 + intensity / 500})`;
  ctx.drawImage(img, 0, 0);
  if (treatment === "combined") { ctx.filter = `brightness(${1 + intensity / 350}) saturate(${1 - intensity / 500})`; ctx.globalAlpha = 0.22; ctx.drawImage(img, 0, 0); }
  URL.revokeObjectURL(source);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export async function simulateSmile(file: File, treatment: Treatment, intensity: number): Promise<{ url: string; demo: boolean }> {
  const form = new FormData(); form.append("file", file); form.append("treatment", treatment); form.append("intensity", String(intensity));
  try {
    const controller = new AbortController(); const timer = window.setTimeout(() => controller.abort(), 7000);
    const response = await fetch(`${API_BASE_URL}/api/simulate`, { method: "POST", body: form, signal: controller.signal }); window.clearTimeout(timer);
    if (!response.ok) throw new Error("Backend returned an error.");
    return { url: URL.createObjectURL(await response.blob()), demo: false };
  } catch {
    return { url: await localPreview(file, treatment, intensity), demo: true };
  }
}
