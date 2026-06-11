import { NavLink } from "react-router-dom";
import {
  BookOpen,
  FlaskConical,
  Fingerprint,
  LayoutDashboard,
  ListChecks,
  LogOut,
  Settings,
  ShieldCheck,
  Trophy,
} from "lucide-react";

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/dataset", label: "Dataset Generator", icon: FlaskConical },
  { to: "/transactions", label: "Transactions", icon: ListChecks },
  { to: "/investigations", label: "Investigations", icon: Fingerprint },
  { to: "/reports", label: "Reports", icon: BookOpen },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col bg-sidebar text-white">
      <div className="flex items-center gap-3 px-6 py-6">
        <ShieldCheck className="h-7 w-7 text-brand" />
        <div>
          <div className="text-lg font-bold leading-tight">FraudLab AI</div>
          <div className="text-[10px] uppercase tracking-widest text-slate-400">
            Risk Intelligence
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "sidebar-link-active" : ""}`
            }
          >
            <Icon className="h-5 w-5" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mx-3 mb-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-slate-400">
          <Trophy className="h-3.5 w-3.5" /> Champion Model
        </div>
        <div className="mt-1 font-semibold">XGBoost</div>
      </div>

      <div className="space-y-1 px-3 pb-5">
        <a className="sidebar-link" href="#">
          <BookOpen className="h-5 w-5" /> Documentation
        </a>
        <a className="sidebar-link" href="#">
          <LogOut className="h-5 w-5" /> Logout
        </a>
      </div>
    </aside>
  );
}
