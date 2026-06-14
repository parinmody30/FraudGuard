import type { ReactNode } from "react";

// Dark grid backdrop with a soft brand-blue glow — shared by the landing,
// login and signup pages so they read as one consistent "before app" theme.
export function GridBackground({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-grid bg-grid-fade text-white">
      <div className="relative z-10 flex min-h-screen flex-col">{children}</div>
    </div>
  );
}
