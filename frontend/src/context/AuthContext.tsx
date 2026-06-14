// Lightweight client-side auth (Phase 7.0 contract has no auth endpoints).
// Persists a small "directory" of registered users + the active session in
// localStorage purely to gate the app shell behind login/signup screens.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export interface AuthUser {
  name: string;
  email: string;
}

interface StoredUser extends AuthUser {
  password: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const USERS_KEY = "fraudlab_users";
const SESSION_KEY = "fraudlab_session";

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readUsers(): StoredUser[] {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY) ?? "[]") as StoredUser[];
  } catch {
    return [];
  }
}

function writeUsers(users: StoredUser[]) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (raw) setUser(JSON.parse(raw) as AuthUser);
    } catch {
      // ignore corrupted session
    }
    setReady(true);
  }, []);

  const login = async (email: string, password: string) => {
    const normalized = email.trim().toLowerCase();
    const match = readUsers().find((u) => u.email === normalized && u.password === password);
    if (!match) throw new Error("Invalid email or password.");
    const session: AuthUser = { name: match.name, email: match.email };
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    setUser(session);
  };

  const signup = async (name: string, email: string, password: string) => {
    const normalized = email.trim().toLowerCase();
    const users = readUsers();
    if (users.some((u) => u.email === normalized)) {
      throw new Error("An account with this email already exists.");
    }
    const next: StoredUser = { name: name.trim(), email: normalized, password };
    writeUsers([...users, next]);
    const session: AuthUser = { name: next.name, email: next.email };
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    setUser(session);
  };

  const logout = () => {
    localStorage.removeItem(SESSION_KEY);
    setUser(null);
  };

  if (!ready) return null;

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: Boolean(user), login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
