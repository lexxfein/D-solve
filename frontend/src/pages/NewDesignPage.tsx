import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ImagePlus,
  LoaderCircle,
  RotateCcw,
  Smile,
  Sparkles,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import { simulateSmile } from "../api";
import { getPatients, getDesigns, saveDesigns, makeId } from "../store";
import type { Treatment } from "../types";

const treatments: {
  id: Treatment;
  label: string;
  description: string;
}[] = [
  {
    id: "whitening",
    label: "Whitening",
    description: "Brighten the detected tooth regions.",
  },
  {
    id: "alignment",
    label: "Alignment",
    description: "Simulate a smoother tooth-row alignment.",
  },
  {
    id: "veneers",
    label: "Veneers",
    description: "Simulate the team's veneer surface workflow.",
  },
  {
    id: "combined",
    label: "Combined",
    description: "Run whitening, alignment and veneer simulation in sequence.",
  },
];

function NewDesignPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [treatment, setTreatment] = useState<Treatment>("whitening");
  const [intensity, setIntensity] = useState(50);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [patientId, setPatientId] = useState(() => getPatients()[0]?.id ?? "");

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [previewUrl, resultUrl]);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);

    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResultUrl(null);
    setDemoMode(false);
    setError(null);
  };

  const removeImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSelectedImage(null);
    setPreviewUrl(null);
    setResultUrl(null);
    setError(null);
  };

  const runSimulation = async () => {
    if (!selectedImage) {
      setError("Upload a smile photograph first.");
      return;
    }

    setProcessing(true);
    setError(null);
    try {
      const result = await simulateSmile(selectedImage, treatment, intensity);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      setResultUrl(result.url);
      setDemoMode(result.demo);
      const patient = getPatients().find(p => p.id === patientId);
      const now = new Date().toISOString();
      const designs = getDesigns();
      const existing = designs.find(d => d.imageData && d.patientId === patientId && d.status === "Draft");
      const record = { id: existing?.id ?? makeId("design"), patientId, patientName: patient?.name ?? "Unassigned patient", treatment, intensity, status: "Completed" as const, imageData: previewUrl ?? undefined, resultData: result.url, createdAt: existing?.createdAt ?? now, updatedAt: now };
      saveDesigns(existing ? designs.map(d => d.id === existing.id ? record : d) : [record, ...designs]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed.");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8faf9] px-4 py-6 sm:px-8 lg:px-12">
      <div className="mx-auto max-w-6xl">
        <Link
          to="/"
          className="mb-8 inline-flex items-center gap-2 text-sm font-medium text-teal-900 transition hover:text-teal-700"
        >
          <ArrowLeft size={18} />
          Back to Dashboard
        </Link>

        <div className="mb-8">
          <div className="mb-3 flex items-center gap-3">
            <div className="rounded-xl bg-teal-100 p-3 text-teal-900">
              <Smile size={24} />
            </div>
            <span className="text-sm font-semibold uppercase tracking-wider text-teal-700">
              New Smile Design
            </span>
          </div>
          <h1 className="text-3xl font-bold text-teal-950">
            Create a New Design
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Upload a patient's frontal smile photograph, choose a treatment
            simulation, and compare the original with the CV-generated result.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-teal-950">
                Smile Photograph
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Use a clear frontal photograph for the team's OpenCV/MediaPipe pipeline.
              </p>
            </div>

            {!previewUrl ? (
              <label
                htmlFor="smile-image"
                className="flex min-h-[390px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-teal-200 bg-teal-50/40 px-6 text-center transition hover:border-teal-500 hover:bg-teal-50"
              >
                <div className="mb-4 rounded-full bg-teal-100 p-4 text-teal-900">
                  <ImagePlus size={32} />
                </div>
                <h3 className="text-base font-semibold text-teal-950">
                  Upload Smile Photograph
                </h3>
                <p className="mt-2 text-sm text-slate-500">
                  Click to select an image from your computer
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  PNG, JPG or JPEG • Maximum 10 MB
                </p>
                <span className="mt-5 inline-flex items-center gap-2 rounded-xl bg-teal-900 px-5 py-3 text-sm font-semibold text-white">
                  <Upload size={17} />
                  Choose Image
                </span>
                <input
                  id="smile-image"
                  type="file"
                  accept="image/png, image/jpeg, image/jpg"
                  onChange={handleImageChange}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="space-y-5">
                <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                  <img
                    src={previewUrl}
                    alt="Selected smile photograph"
                    className="mx-auto max-h-[520px] w-full object-contain"
                  />
                  <button
                    type="button"
                    onClick={removeImage}
                    className="absolute right-4 top-4 rounded-full bg-white p-2 text-slate-700 shadow-md transition hover:bg-red-50 hover:text-red-600"
                    aria-label="Remove selected image"
                  >
                    <X size={20} />
                  </button>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-sm font-semibold text-teal-950">
                    {selectedImage?.name}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {selectedImage
                      ? `${(selectedImage.size / 1024 / 1024).toFixed(2)} MB`
                      : ""}
                  </p>
                </div>

                <label
                  htmlFor="replace-image"
                  className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-teal-200 px-5 py-3 text-sm font-semibold text-teal-900 transition hover:bg-teal-50"
                >
                  <Upload size={17} />
                  Choose Another Image
                  <input
                    id="replace-image"
                    type="file"
                    accept="image/png, image/jpeg, image/jpg"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="mb-5 rounded-xl bg-slate-50 p-4">
              <label className="text-sm font-semibold text-slate-800">Patient</label>
              <select value={patientId} onChange={e => setPatientId(e.target.value)} className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-teal-600">
                <option value="">Unassigned patient</option>
                {getPatients().map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-teal-50 p-3 text-teal-800">
                <WandSparkles size={21} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-teal-950">
                  Treatment Simulation
                </h2>
                <p className="text-sm text-slate-500">
                  Select the CV operation to run.
                </p>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              {treatments.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setTreatment(item.id)}
                  className={`w-full rounded-xl border p-4 text-left transition ${
                    treatment === item.id
                      ? "border-teal-700 bg-teal-50 ring-1 ring-teal-700"
                      : "border-slate-200 hover:border-teal-300 hover:bg-slate-50"
                  }`}
                >
                  <p className="font-semibold text-slate-900">{item.label}</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    {item.description}
                  </p>
                </button>
              ))}
            </div>

            <div className="mt-7">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="intensity"
                  className="text-sm font-semibold text-slate-800"
                >
                  Simulation intensity
                </label>
                <span className="rounded-lg bg-teal-50 px-2.5 py-1 text-sm font-bold text-teal-900">
                  {intensity}%
                </span>
              </div>
              <input
                id="intensity"
                type="range"
                min="0"
                max="100"
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
                className="mt-4 w-full accent-teal-800"
              />
            </div>

            <button
              type="button"
              disabled={!selectedImage || processing}
              onClick={runSimulation}
              className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl bg-teal-900 px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-teal-950 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {processing ? (
                <>
                  <LoaderCircle className="animate-spin" size={18} />
                  Processing image...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Generate Smile Simulation
                </>
              )}
            </button>

            {demoMode && resultUrl && (
              <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm leading-6 text-blue-800">Backend is not connected, so this is a frontend demo preview. Your CV code has not been changed. Once the backend is available, the same button will use the real CV pipeline.
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">
                {error}
              </div>
            )}

            <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-800">
              This is a visual simulation for treatment communication, not a
              clinical prediction or treatment recommendation.
            </div>
          </section>
        </div>

        {previewUrl && (
          <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-teal-950">
                  Before & After
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  {resultUrl
                    ? `${treatments.find((t) => t.id === treatment)?.label} simulation at ${intensity}% intensity`
                    : "Generate a simulation to see the processed result."}
                </p>
              </div>

              {resultUrl && (
                <button
                  type="button"
                  onClick={runSimulation}
                  disabled={processing}
                  className="inline-flex items-center gap-2 rounded-xl border border-teal-200 px-4 py-2.5 text-sm font-semibold text-teal-900 hover:bg-teal-50 disabled:opacity-50"
                >
                  <RotateCcw size={16} />
                  Re-run
                </button>
              )}
            </div>

            <div className="mt-6 grid gap-5 md:grid-cols-2">
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Original
                </p>
                <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
                  <img
                    src={previewUrl}
                    alt="Original smile"
                    className="max-h-[560px] w-full object-contain"
                  />
                </div>
              </div>

              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-teal-700">
                  Simulated
                </p>
                <div className="flex min-h-[250px] items-center justify-center overflow-hidden rounded-xl border border-teal-200 bg-slate-100">
                  {resultUrl ? (
                    <img
                      src={resultUrl}
                      alt="Simulated smile treatment result"
                      className="max-h-[560px] w-full object-contain"
                    />
                  ) : (
                    <div className="px-8 text-center">
                      <Smile className="mx-auto text-teal-700" size={34} />
                      <p className="mt-3 text-sm font-medium text-slate-700">
                        Your simulation will appear here.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default NewDesignPage;
