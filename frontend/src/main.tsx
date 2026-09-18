import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import DashboardPage from "./pages/DashboardPage";
import NewDesignPage from "./pages/NewDesignPage";
import PatientsPage from "./pages/PatientsPage";
import DesignsPage from "./pages/DesignsPage";
import ReportsPage from "./pages/ReportsPage";
import SettingsPage from "./pages/SettingsPage";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<React.StrictMode><BrowserRouter><Routes><Route element={<App />}><Route path="/" element={<DashboardPage/>}/><Route path="/new-design" element={<NewDesignPage/>}/><Route path="/designs" element={<DesignsPage/>}/><Route path="/patients" element={<PatientsPage/>}/><Route path="/reports" element={<ReportsPage/>}/><Route path="/settings" element={<SettingsPage/>}/></Route></Routes></BrowserRouter></React.StrictMode>);
