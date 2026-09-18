import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  Box,
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
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { simulateSmile, autoAlignModel, getModelInfo, transformModel } from "../api";
import { getPatients, getDesigns, saveDesigns, makeId } from "../store";
import type { Treatment } from "../types";

const twoDTreatments: { id: Exclude<Treatment, "alignment">; label: string; description: string }[] = [
  { id: "whitening", label: "Whitening", description: "Brighten the detected tooth regions." },
  { id: "veneers", label: "Veneers", description: "Simulate the team's existing veneer surface workflow." },
  { id: "combined", label: "Combined", description: "Run the existing 2D whitening and veneer workflows together." },
];

type TransformValues = { tx: number; ty: number; tz: number; rx: number; ry: number; rz: number; scale: number };

const initialTransform: TransformValues = { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, scale: 1 };

function ModelViewer({ url, format = "glb", className = "" }: { url: string | null; format?: "glb" | "gltf" | "obj" | "stl"; className?: string }) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#020718");

    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 1000);
    camera.position.set(0, 0.25, 4.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.4;
    controls.maxDistance = 30;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 3.0);
    key.position.set(3, 5, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1.4);
    fill.position.set(-4, 2, 3);
    scene.add(fill);

    const grid = new THREE.GridHelper(7, 28, 0x20304d, 0x122038);
    grid.position.y = -1.25;
    scene.add(grid);

    const resize = () => {
      const width = mount.clientWidth || 600;
      const height = mount.clientHeight || 420;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    let cancelled = false;
    const load = async () => {
      if (!url) return;
      try {
        let object: THREE.Object3D;
        if (format === "obj") {
          const text = await fetch(url).then(r => r.text());
          object = new OBJLoader().parse(text);
          object.traverse(child => {
            const mesh = child as THREE.Mesh;
            if (mesh.isMesh && !mesh.material) mesh.material = new THREE.MeshStandardMaterial({ color: 0xf4ead4 });
          });
        } else if (format === "stl") {
          const buffer = await fetch(url).then(r => r.arrayBuffer());
          const geometry = new STLLoader().parse(buffer);
          geometry.computeVertexNormals();
          object = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xf4ead4, roughness: 0.55, metalness: 0.02 }));
        } else {
          const gltf = await new GLTFLoader().loadAsync(url);
          object = gltf.scene;
        }
        if (cancelled) return;

        object.traverse(child => {
          const mesh = child as THREE.Mesh;
          if (mesh.isMesh) {
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            materials.forEach(material => {
              const standard = material as THREE.MeshStandardMaterial;
              if (standard.roughness !== undefined) standard.roughness = Math.max(0.35, standard.roughness);
            });
          }
        });

        const box = new THREE.Box3().setFromObject(object);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 0.001);
        const scale = 2.4 / maxDim;
        object.position.sub(center);
        object.scale.setScalar(scale);
        object.position.y += 0.05;
        scene.add(object);

        const fitted = new THREE.Box3().setFromObject(object);
        const fittedSize = fitted.getSize(new THREE.Vector3());
        const distance = Math.max(fittedSize.x, fittedSize.y, fittedSize.z) * 1.7;
        camera.position.set(0, fittedSize.y * 0.08, Math.max(distance, 3.0));
        controls.target.set(0, 0, 0);
        controls.update();
      } catch {
        // The parent page displays the upload/backend error; the viewer stays empty.
      }
    };
    load();

    const animate = () => {
      if (cancelled) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelled = true;
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse(object => {
        const mesh = object as THREE.Mesh;
        if (mesh.isMesh) {
          mesh.geometry?.dispose();
          const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          materials.forEach(material => material.dispose());
        }
      });
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, [url, format]);

  return <div ref={mountRef} className={`min-h-[360px] w-full overflow-hidden rounded-2xl ${className}`} />;
}

function NewDesignPage() {
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [treatment, setTreatment] = useState<Exclude<Treatment, "alignment">>("whitening");
  const [intensity, setIntensity] = useState(50);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [patientId, setPatientId] = useState(() => getPatients()[0]?.id ?? "");

  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [alignedUrl, setAlignedUrl] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<{ vertices: number; faces: number; meshes: number; size: number[] } | null>(null);
  const [alignmentStrength, setAlignmentStrength] = useState(100);
  const [transform, setTransform] = useState<TransformValues>(initialTransform);
  const [autoSummary, setAutoSummary] = useState<string | null>(null);

  useEffect(() => () => {
    [previewUrl, resultUrl, modelUrl, alignedUrl].forEach(url => { if (url) URL.revokeObjectURL(url); });
  }, [previewUrl, resultUrl, modelUrl, alignedUrl]);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) { setError("Please select a valid image file."); return; }
    if (file.size > 10 * 1024 * 1024) { setError("Image size must be less than 10 MB."); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSelectedImage(file); setPreviewUrl(URL.createObjectURL(file)); setResultUrl(null); setDemoMode(false); setError(null);
  };

  const removeImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSelectedImage(null); setPreviewUrl(null); setResultUrl(null); setError(null);
  };

  const handleModelChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const ext = file.name.toLowerCase().split(".").pop();
    if (!ext || !["glb", "gltf", "obj", "stl"].includes(ext)) {
      setError("Please select a GLB, GLTF, OBJ or STL dental model."); return;
    }
    if (file.size > 80 * 1024 * 1024) { setError("3D model must be less than 80 MB."); return; }
    if (modelUrl) URL.revokeObjectURL(modelUrl);
    if (alignedUrl) URL.revokeObjectURL(alignedUrl);
    setModelFile(file); setModelUrl(URL.createObjectURL(file)); setAlignedUrl(null); setTransform({ ...initialTransform }); setAutoSummary(null); setModelInfo(null); setError(null);
    try {
      const info = await getModelInfo(file);
      setModelInfo(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze the 3D model. Make sure the backend is running.");
    }
  };

  const runSimulation = async () => {
    if (!selectedImage) { setError("Upload a smile photograph first."); return; }
    setProcessing(true); setError(null);
    try {
      const result = await simulateSmile(selectedImage, treatment, intensity);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      setResultUrl(result.url); setDemoMode(result.demo);
      const patient = getPatients().find(p => p.id === patientId);
      const now = new Date().toISOString(); const designs = getDesigns();
      const existing = designs.find(d => d.imageData && d.patientId === patientId && d.status === "Draft");
      const record = { id: existing?.id ?? makeId("design"), patientId, patientName: patient?.name ?? "Unassigned patient", treatment, intensity, status: "Completed" as const, imageData: previewUrl ?? undefined, resultData: result.url, createdAt: existing?.createdAt ?? now, updatedAt: now };
      saveDesigns(existing ? designs.map(d => d.id === existing.id ? record : d) : [record, ...designs]);
    } catch (err) { setError(err instanceof Error ? err.message : "Simulation failed."); }
    finally { setProcessing(false); }
  };

  const applyAutoAlignment = async () => {
    if (!modelFile) { setError("Upload a 3D dental model first."); return; }
    setProcessing(true); setError(null);
    try {
      const result = await autoAlignModel(modelFile, alignmentStrength);
      if (alignedUrl) URL.revokeObjectURL(alignedUrl);
      const url = URL.createObjectURL(result.blob);
      setAlignedUrl(url);
      setTransform({
        tx: Number(result.transform.translation?.[0] ?? 0),
        ty: Number(result.transform.translation?.[1] ?? 0),
        tz: Number(result.transform.translation?.[2] ?? 0),
        rx: Number(result.transform.rotation?.[0] ?? 0),
        ry: Number(result.transform.rotation?.[1] ?? 0),
        rz: Number(result.transform.rotation?.[2] ?? 0),
        scale: Number(result.transform.scale ?? 1),
      });
      setAutoSummary(result.transform.note || `Automatic dental-arch alignment applied at ${alignmentStrength}%.`);
    } catch (err) { setError(err instanceof Error ? err.message : "Automatic 3D alignment failed."); }
    finally { setProcessing(false); }
  };

  const applyManualTransform = async () => {
    if (!modelFile) { setError("Upload a 3D dental model first."); return; }
    setProcessing(true); setError(null);
    try {
      const blob = await transformModel(modelFile, transform);
      if (alignedUrl) URL.revokeObjectURL(alignedUrl);
      setAlignedUrl(URL.createObjectURL(blob));
      setAutoSummary("Manual 3D transform exported from the backend.");
    } catch (err) { setError(err instanceof Error ? err.message : "Could not apply the 3D transform."); }
    finally { setProcessing(false); }
  };

  const reset3D = () => {
    setTransform({ ...initialTransform }); setAutoSummary(null);
    if (alignedUrl) URL.revokeObjectURL(alignedUrl);
    setAlignedUrl(null); setError(null);
  };

  const updateTransform = (key: keyof TransformValues, value: string) => {
    const numeric = Number(value);
    setTransform(prev => ({ ...prev, [key]: Number.isFinite(numeric) ? numeric : prev[key] }));
  };

  return (
    <div className="min-h-screen bg-[#f8faf9] px-4 py-6 sm:px-8 lg:px-12">
      <div className="mx-auto max-w-6xl">
        <Link to="/" className="mb-8 inline-flex items-center gap-2 text-sm font-medium text-teal-900 transition hover:text-teal-700"><ArrowLeft size={18} />Back to Dashboard</Link>

        <div className="mb-7">
          <div className="mb-3 flex items-center gap-3"><div className="rounded-xl bg-teal-100 p-3 text-teal-900"><Smile size={24} /></div><span className="text-sm font-semibold uppercase tracking-wider text-teal-700">New Smile Design</span></div>
          <h1 className="text-3xl font-bold text-teal-950">Create a New Design</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">2D photographs use the existing Whitening and Veneer CV workflows. 3D Alignment is a separate workflow for uploaded dental meshes.</p>
        </div>

        <div className="mb-6 grid grid-cols-2 rounded-2xl border border-slate-200 bg-white p-1.5 shadow-sm">
          <button onClick={() => { setMode("2d"); setError(null); }} className={`rounded-xl px-4 py-3 text-sm font-semibold ${mode === "2d" ? "bg-teal-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}><span className="inline-flex items-center gap-2"><ImagePlus size={17} />2D Smile Design</span></button>
          <button onClick={() => { setMode("3d"); setError(null); }} className={`rounded-xl px-4 py-3 text-sm font-semibold ${mode === "3d" ? "bg-teal-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}><span className="inline-flex items-center gap-2"><Box size={17} />3D Alignment</span></button>
        </div>

        {mode === "2d" ? (
          <>
            <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                <div className="mb-6"><h2 className="text-lg font-semibold text-teal-950">Smile Photograph</h2><p className="mt-1 text-sm text-slate-500">Use a clear frontal photograph for the existing OpenCV/MediaPipe pipeline.</p></div>
                {!previewUrl ? (
                  <label htmlFor="smile-image" className="flex min-h-[390px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-teal-200 bg-teal-50/40 px-6 text-center transition hover:border-teal-500 hover:bg-teal-50">
                    <div className="mb-4 rounded-full bg-teal-100 p-4 text-teal-900"><ImagePlus size={32} /></div><h3 className="text-base font-semibold text-teal-950">Upload Smile Photograph</h3><p className="mt-2 text-sm text-slate-500">Click to select an image from your computer</p><p className="mt-1 text-xs text-slate-400">PNG, JPG or JPEG • Maximum 10 MB</p><span className="mt-5 inline-flex items-center gap-2 rounded-xl bg-teal-900 px-5 py-3 text-sm font-semibold text-white"><Upload size={17} />Choose Image</span>
                    <input id="smile-image" type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleImageChange} className="hidden" />
                  </label>
                ) : (
                  <div className="space-y-5"><div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-100"><img src={previewUrl} alt="Selected smile photograph" className="mx-auto max-h-[520px] w-full object-contain" /><button type="button" onClick={removeImage} className="absolute right-4 top-4 rounded-full bg-white p-2 text-slate-700 shadow-md hover:bg-red-50 hover:text-red-600"><X size={20} /></button></div><div className="rounded-xl bg-slate-50 p-4"><p className="text-sm font-semibold text-teal-950">{selectedImage?.name}</p><p className="mt-1 text-xs text-slate-500">{selectedImage ? `${(selectedImage.size / 1024 / 1024).toFixed(2)} MB` : ""}</p></div><label htmlFor="replace-image" className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-teal-200 px-5 py-3 text-sm font-semibold text-teal-900 hover:bg-teal-50"><Upload size={17} />Choose Another Image<input id="replace-image" type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleImageChange} className="hidden" /></label></div>
                )}
              </section>

              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                <div className="mb-5 rounded-xl bg-slate-50 p-4"><label className="text-sm font-semibold text-slate-800">Patient</label><select value={patientId} onChange={e => setPatientId(e.target.value)} className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-teal-600"><option value="">Unassigned patient</option>{getPatients().map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select></div>
                <div className="flex items-center gap-3"><div className="rounded-xl bg-teal-50 p-3 text-teal-800"><WandSparkles size={21} /></div><div><h2 className="text-lg font-semibold text-teal-950">Treatment Simulation</h2><p className="text-sm text-slate-500">Select the existing 2D CV operation.</p></div></div>
                <div className="mt-6 space-y-3">{twoDTreatments.map(item => <button key={item.id} type="button" onClick={() => setTreatment(item.id)} className={`w-full rounded-xl border p-4 text-left transition ${treatment === item.id ? "border-teal-700 bg-teal-50 ring-1 ring-teal-700" : "border-slate-200 hover:border-teal-300 hover:bg-slate-50"}`}><p className="font-semibold text-slate-900">{item.label}</p><p className="mt-1 text-xs leading-5 text-slate-500">{item.description}</p></button>)}</div>
                <div className="mt-7"><div className="flex items-center justify-between"><label htmlFor="intensity" className="text-sm font-semibold text-slate-800">Simulation intensity</label><span className="rounded-lg bg-teal-50 px-2.5 py-1 text-sm font-bold text-teal-900">{intensity}%</span></div><input id="intensity" type="range" min="0" max="100" value={intensity} onChange={e => setIntensity(Number(e.target.value))} className="mt-4 w-full accent-teal-800" /></div>
                <button type="button" disabled={!selectedImage || processing} onClick={runSimulation} className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl bg-teal-900 px-5 py-3.5 text-sm font-semibold text-white hover:bg-teal-950 disabled:cursor-not-allowed disabled:opacity-50">{processing ? <><LoaderCircle className="animate-spin" size={18} />Processing image...</> : <><Sparkles size={18} />Generate Smile Simulation</>}</button>
                {demoMode && resultUrl && <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm leading-6 text-blue-800">Backend is not connected, so this is only the existing frontend demo preview. Whitening/Veneers CV are unchanged.</div>}
                {error && <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">{error}</div>}
                <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-800">This is a visual simulation for treatment communication, not a clinical prediction or treatment recommendation.</div>
              </section>
            </div>
            {previewUrl && <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8"><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold text-teal-950">Before & After</h2><p className="mt-1 text-sm text-slate-500">{resultUrl ? `${twoDTreatments.find(t => t.id === treatment)?.label} simulation at ${intensity}% intensity` : "Generate a simulation to see the processed result."}</p></div>{resultUrl && <button type="button" onClick={runSimulation} disabled={processing} className="inline-flex items-center gap-2 rounded-xl border border-teal-200 px-4 py-2.5 text-sm font-semibold text-teal-900 hover:bg-teal-50"><RotateCcw size={16} />Re-run</button>}</div><div className="mt-6 grid gap-5 md:grid-cols-2"><div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Original</p><div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-100"><img src={previewUrl} alt="Original smile" className="max-h-[560px] w-full object-contain" /></div></div><div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-teal-700">Simulated</p><div className="flex min-h-[250px] items-center justify-center overflow-hidden rounded-xl border border-teal-200 bg-slate-100">{resultUrl ? <img src={resultUrl} alt="Simulated smile treatment result" className="max-h-[560px] w-full object-contain" /> : <div className="px-8 text-center"><Smile className="mx-auto text-teal-700" size={34} /><p className="mt-3 text-sm font-medium text-slate-700">Your simulation will appear here.</p></div>}</div></div></div></section>}
          </>
        ) : (
          <>
            <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                <div className="mb-5 flex items-start justify-between gap-4"><div><h2 className="text-lg font-semibold text-teal-950">3D Dental Model</h2><p className="mt-1 text-sm leading-6 text-slate-500">Upload the dental scan/model you want to orient and align in 3D.</p></div><div className="rounded-xl bg-teal-50 p-3 text-teal-800"><Box size={22} /></div></div>
                {!modelUrl ? <label htmlFor="model-file" className="flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-teal-200 bg-teal-50/40 px-6 text-center hover:border-teal-500 hover:bg-teal-50"><div className="mb-4 rounded-full bg-teal-100 p-4 text-teal-900"><Box size={30} /></div><h3 className="font-semibold text-teal-950">Upload 3D Model</h3><p className="mt-2 text-sm text-slate-500">GLB, GLTF, OBJ or STL • Maximum 80 MB</p><span className="mt-5 inline-flex items-center gap-2 rounded-xl bg-teal-900 px-5 py-3 text-sm font-semibold text-white"><Upload size={17} />Choose 3D File</span><input id="model-file" type="file" accept=".glb,.gltf,.obj,.stl,model/gltf-binary,model/gltf+json" onChange={handleModelChange} className="hidden" /></label> : <div className="space-y-4"><ModelViewer url={modelUrl} format={(modelFile?.name.toLowerCase().split(".").pop() as "glb" | "gltf" | "obj" | "stl") || "glb"} /><div className="flex items-center justify-between rounded-xl bg-slate-50 p-4"><div><p className="text-sm font-semibold text-teal-950">{modelFile?.name}</p><p className="mt-1 text-xs text-slate-500">{modelFile ? `${(modelFile.size / 1024 / 1024).toFixed(2)} MB` : ""}</p></div><label htmlFor="replace-model" className="cursor-pointer rounded-lg border border-teal-200 px-3 py-2 text-xs font-semibold text-teal-900 hover:bg-teal-50">Replace<input id="replace-model" type="file" accept=".glb,.gltf,.obj,.stl,model/gltf-binary,model/gltf+json" onChange={handleModelChange} className="hidden" /></label></div>{modelInfo && <div className="grid grid-cols-3 gap-2 text-center"><div className="rounded-xl bg-slate-50 p-3"><p className="text-xs text-slate-500">Meshes</p><p className="mt-1 font-bold text-teal-950">{modelInfo.meshes}</p></div><div className="rounded-xl bg-slate-50 p-3"><p className="text-xs text-slate-500">Vertices</p><p className="mt-1 font-bold text-teal-950">{modelInfo.vertices.toLocaleString()}</p></div><div className="rounded-xl bg-slate-50 p-3"><p className="text-xs text-slate-500">Faces</p><p className="mt-1 font-bold text-teal-950">{modelInfo.faces.toLocaleString()}</p></div></div>}</div>}
              </section>

              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                <div className="mb-6"><h2 className="text-lg font-semibold text-teal-950">3D Alignment Controls</h2><p className="mt-1 text-sm leading-6 text-slate-500">Auto Align normalizes the dental arch shape of the uploaded fused scan. Manual controls apply a rigid transform to the complete model.</p></div>
                <div className="rounded-xl bg-slate-50 p-4"><div className="flex items-center justify-between"><label htmlFor="alignment-strength" className="text-sm font-semibold text-slate-800">Automatic alignment strength</label><span className="font-bold text-teal-950">{alignmentStrength}%</span></div><input id="alignment-strength" type="range" min="0" max="100" value={alignmentStrength} onChange={e => setAlignmentStrength(Number(e.target.value))} className="mt-4 w-full accent-teal-800" /><button type="button" disabled={!modelFile || processing} onClick={applyAutoAlignment} className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-teal-900 px-5 py-3 text-sm font-semibold text-white hover:bg-teal-950 disabled:cursor-not-allowed disabled:opacity-50">{processing ? <LoaderCircle className="animate-spin" size={18} /> : <Sparkles size={17} />}Auto Align Model</button></div>

                <div className="mt-6 grid grid-cols-2 gap-3">{(["tx", "ty", "tz", "rx", "ry", "rz"] as const).map(key => <label key={key} className="block"><span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{key[0] === "t" ? `Translation ${key[1].toUpperCase()}` : `Rotation ${key[1].toUpperCase()}`} {key[0] === "r" && "(°)"}</span><input type="number" step={key[0] === "r" ? "0.1" : "0.01"} value={transform[key]} onChange={e => updateTransform(key, e.target.value)} className="mt-1.5 w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-teal-600" /></label>)}</div>
                <label className="mt-3 block"><span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Scale (1 = original size)</span><input type="number" min="0.01" step="0.01" value={transform.scale} onChange={e => updateTransform("scale", e.target.value)} className="mt-1.5 w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-teal-600" /></label>
                <div className="mt-5 grid grid-cols-2 gap-3"><button type="button" onClick={reset3D} className="inline-flex items-center justify-center gap-2 rounded-xl border border-teal-200 px-4 py-3 text-sm font-semibold text-teal-900 hover:bg-teal-50"><RotateCcw size={16} />Reset</button><button type="button" disabled={!modelFile || processing} onClick={applyManualTransform} className="inline-flex items-center justify-center gap-2 rounded-xl bg-teal-900 px-4 py-3 text-sm font-semibold text-white hover:bg-teal-950 disabled:opacity-50"><WandSparkles size={16} />Apply Transform</button></div>
                {autoSummary && <div className="mt-5 rounded-xl border border-teal-200 bg-teal-50 p-4 text-xs leading-5 text-teal-900">{autoSummary}</div>}
                {error && <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">{error}</div>}
                <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-800"><strong>Important:</strong> Auto Align now applies a visible, smooth dental-arch normalization to this fused scan. It is a visual design transformation, not individual tooth movement or a clinical orthodontic prediction. Individual tooth movement requires segmented tooth objects.</div>
              </section>
            </div>

            {modelUrl && <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8"><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold text-teal-950">Before & After 3D</h2><p className="mt-1 text-sm text-slate-500">The left side is the uploaded model. The right side is the exported transformed GLB.</p></div>{alignedUrl && <a href={alignedUrl} download="dsolve-aligned.glb" className="inline-flex items-center gap-2 rounded-xl border border-teal-200 px-4 py-2.5 text-sm font-semibold text-teal-900 hover:bg-teal-50"><Box size={16} />Export GLB</a>}</div><div className="mt-6 grid gap-5 md:grid-cols-2"><div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Original 3D Model</p><div className="overflow-hidden rounded-2xl border border-slate-200"><ModelViewer url={modelUrl} format={(modelFile?.name.toLowerCase().split(".").pop() as "glb" | "gltf" | "obj" | "stl") || "glb"} /></div></div><div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-teal-700">Aligned 3D Result</p><div className="overflow-hidden rounded-2xl border border-teal-200">{alignedUrl ? <ModelViewer url={alignedUrl} format="glb" /> : <div className="flex min-h-[360px] items-center justify-center bg-[#020718] px-8 text-center text-white"><div><Box className="mx-auto opacity-70" size={36} /><p className="mt-3 text-sm font-medium">Run Auto Align or Apply Transform to create the exported result.</p></div></div>}</div></div></div></section>}
          </>
        )}
      </div>
    </div>
  );
}

export default NewDesignPage;
