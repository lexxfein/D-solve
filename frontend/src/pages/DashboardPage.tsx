import { Activity, ArrowRight, CalendarDays, FolderOpen, Plus, Smile, Users } from "lucide-react";
import { Link } from "react-router-dom";
import { getDesigns, getPatients } from "../store";
import { useEffect, useState } from "react";
import type { SmileDesign } from "../types";

export default function DashboardPage() {
  const [designs,setDesigns]=useState<SmileDesign[]>([]); const [patients,setPatients]=useState(0);
  useEffect(()=>{setDesigns(getDesigns());setPatients(getPatients().length)},[]);
  const completed=designs.filter(d=>d.status==="Completed").length;
  const stats=[{label:"Total Cases",value:designs.length,icon:FolderOpen},{label:"Active Designs",value:designs.filter(d=>d.status!=="Completed").length,icon:Smile},{label:"Completed",value:completed,icon:Activity},{label:"Patients",value:patients,icon:Users}];
  return <div className="space-y-8 px-5 py-8 md:px-10">
    <section><p className="text-sm font-medium text-teal-700">Welcome back</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">Design better smiles.</h1><p className="mt-2 max-w-2xl text-slate-500">Manage patients, create visual treatment simulations, and keep every smile design case organized in one place.</p></section>
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{stats.map(({label,value,icon:Icon})=><div key={label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="rounded-xl bg-teal-50 p-3 text-teal-800 w-fit"><Icon size={21}/></div><p className="mt-4 text-sm text-slate-500">{label}</p><p className="mt-1 text-3xl font-bold text-slate-900">{String(value).padStart(2,"0")}</p></div>)}</section>
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="flex items-center justify-between"><div><h2 className="font-bold text-slate-900">Recent Smile Designs</h2><p className="mt-1 text-sm text-slate-500">Your latest cases</p></div><Link to="/designs" className="text-sm font-semibold text-teal-800">View all <ArrowRight className="inline" size={16}/></Link></div>
      {designs.length===0?<div className="mt-6 flex flex-col items-center rounded-xl border border-dashed border-slate-300 px-6 py-12 text-center"><CalendarDays className="text-teal-700" size={30}/><h3 className="mt-4 font-semibold">No design cases yet</h3><p className="mt-2 text-sm text-slate-500">Create your first case to start exploring treatment visualizations.</p><Link to="/new-design" className="mt-5 flex items-center gap-2 rounded-xl bg-teal-900 px-5 py-3 text-sm font-semibold text-white"><Plus size={17}/> Create First Design</Link></div>:<div className="mt-5 divide-y divide-slate-100">{designs.slice(0,5).map(d=><Link key={d.id} to={`/designs?case=${d.id}`} className="flex items-center justify-between gap-4 py-4 hover:bg-slate-50"><div><p className="font-semibold text-slate-800">{d.patientName}</p><p className="text-xs text-slate-500">{d.treatment} · {new Date(d.updatedAt).toLocaleDateString()}</p></div><span className="rounded-full bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-800">{d.status}</span></Link>)}</div>}
    </section>
  </div>;
}
