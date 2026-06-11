import { EmptyState } from "@/components/common/States";

export default function Placeholder({ title }: { title: string }) {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-900">{title}</h1>
      <EmptyState message={`${title} is part of the navigation shell. No backend endpoint is defined for it in the Phase 7.0 contract.`} />
    </div>
  );
}
