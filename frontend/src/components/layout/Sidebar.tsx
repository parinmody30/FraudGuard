import { NavLink, useNavigate } from "react-router-dom";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Fingerprint,
  LayoutDashboard,
  ListChecks,
  LogOut,
  Settings,
  ShieldCheck,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/dataset", label: "Dataset Generator", icon: FlaskConical },
  { to: "/transactions", label: "Transactions", icon: ListChecks },
  { to: "/investigations", label: "Investigations", icon: Fingerprint },
  { to: "/reports", label: "Reports", icon: BookOpen },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({
  collapsed,
  onToggleCollapsed,
}: {
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <aside
      className={`relative flex h-screen shrink-0 flex-col bg-sidebar text-white transition-all duration-200 ${
        collapsed ? "w-20" : "w-64"
      }`}
    >
      <div className="flex items-center gap-3 overflow-hidden px-6 py-6">
        <ShieldCheck className="h-7 w-7 shrink-0 text-brand" />
        {!collapsed && (
          <div className="overflow-hidden">
            <div className="truncate text-lg font-bold leading-tight">FraudLab AI</div>
            <div className="truncate text-[10px] uppercase tracking-widest text-slate-400">
              Risk Intelligence
            </div>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        <button
          onClick={onToggleCollapsed}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={`sidebar-link w-full ${collapsed ? "justify-center px-0" : ""}`}
        >
          {collapsed ? <ChevronRight className="h-5 w-5 shrink-0" /> : <ChevronLeft className="h-5 w-5 shrink-0" />}
          {!collapsed && <span className="truncate">Collapse</span>}
        </button>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            title={collapsed ? label : undefined}
            className={({ isActive }) =>
              `sidebar-link ${collapsed ? "justify-center px-0" : ""} ${isActive ? "sidebar-link-active" : ""}`
            }
          >
            <Icon className="h-5 w-5 shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {user && (
        <div
          className={`mx-3 mb-3 truncate rounded-xl border border-white/10 bg-white/5 text-xs text-slate-300 ${
            collapsed ? "px-2 py-2 text-center" : "px-4 py-2"
          }`}
          title={collapsed ? `Signed in as ${user.name || user.email}` : undefined}
        >
          {collapsed ? (
            <span className="font-semibold text-white">{(user.name || user.email)[0]?.toUpperCase()}</span>
          ) : (
            <>
              Signed in as <span className="font-semibold text-white">{user.name || user.email}</span>
            </>
          )}
        </div>
      )}

      <div className="space-y-1 px-3 pb-5">
        <a
          className={`sidebar-link ${collapsed ? "justify-center px-0" : ""}`}
          href="#"
          title={collapsed ? "Documentation" : undefined}
        >
          <BookOpen className="h-5 w-5 shrink-0" /> {!collapsed && "Documentation"}
        </a>
        <button
          className={`sidebar-link w-full ${collapsed ? "justify-center px-0" : ""}`}
          title={collapsed ? "Logout" : undefined}
          onClick={() => {
            logout();
            navigate("/login", { replace: true });
          }}
        >
          <LogOut className="h-5 w-5 shrink-0" /> {!collapsed && "Logout"}
        </button>
      </div>
    </aside>
  );
}
