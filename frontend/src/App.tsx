import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Activity, FileText, LayoutDashboard, Plus, Settings, Smile, Users, Menu, X } from "lucide-react";
import { useState } from "react";

const items = [
  { label: "Dashboard", path: "/", icon: LayoutDashboard },
  { label: "Smile Designs", path: "/designs", icon: Smile },
  { label: "Patients", path: "/patients", icon: Users },
  { label: "Reports", path: "/reports", icon: FileText },
  { label: "Settings", path: "/settings", icon: Settings },
];

function App() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const title = items.find(i => i.path === location.pathname)?.label ?? (location.pathname === "/new-design" ? "New Smile Design" : "Dentaform");

  return <div className="min-h-screen bg-[#f7f9f8] text-slate-800">
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-slate-200 bg-white px-5 py-6 transition-transform md:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-900 text-white"><Smile size={23}/></div>
          <div><h1 className="font-bold text-teal-950">Dentaform</h1><p className="text-xs text-slate-500">Smile Design Studio</p></div>
        </div>
        <button className="md:hidden" onClick={() => setOpen(false)}><X size={20}/></button>
      </div>
      <nav className="mt-10 flex-1 space-y-2">
        {items.map(({label,path,icon:Icon}) => <NavLink key={path} to={path} end={path === "/"} onClick={() => setOpen(false)} className={({isActive}) => `flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${isActive ? "bg-teal-900 text-white" : "text-slate-600 hover:bg-teal-50 hover:text-teal-900"}`}><Icon size={19}/>{label}</NavLink>)}
      </nav>
      <button onClick={() => navigate("/new-design")} className="flex items-center justify-center gap-2 rounded-xl bg-teal-900 px-4 py-3 text-sm font-semibold text-white hover:bg-teal-950"><Plus size={18}/> New Design</button>
    </aside>

    {open && <button aria-label="Close menu" className="fixed inset-0 z-30 bg-slate-900/20 md:hidden" onClick={() => setOpen(false)}/>} 
    <main className="min-h-screen md:ml-64">
      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-slate-200 bg-white/95 px-5 py-4 backdrop-blur md:px-10">
        <div className="flex items-center gap-3"><button className="rounded-lg p-2 hover:bg-slate-100 md:hidden" onClick={() => setOpen(true)}><Menu size={21}/></button><div><p className="text-xs text-slate-500">Digital Smile Design</p><h2 className="text-xl font-bold text-teal-950">{title}</h2></div></div>
        <button onClick={() => navigate("/new-design")} className="hidden items-center gap-2 rounded-xl bg-teal-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-teal-950 sm:flex"><Plus size={17}/> New Design</button>
      </header>
      <Outlet />
    </main>
  </div>;
}
export default App;
