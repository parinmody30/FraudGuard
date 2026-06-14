import { Link } from "react-router-dom";
import {
  ArrowRight,
  Fingerprint,
  Gauge,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";
import { GridBackground } from "@/components/layout/GridBackground";

const FEATURES = [
  {
    icon: ShieldCheck,
    title: "Real-time Scoring",
    description: "Score transactions against the XGBoost champion model with calibrated fraud probabilities and decisions.",
  },
  {
    icon: Sparkles,
    title: "Explainable AI",
    description: "SHAP-driven contributor breakdowns show exactly why a transaction was flagged, increase or decrease.",
  },
  {
    icon: Fingerprint,
    title: "AI Investigations",
    description: "Generate investigation summaries, detected signals and next steps for analysts in seconds.",
  },
  {
    icon: Workflow,
    title: "Synthetic Datasets",
    description: "Spin up realistic transaction datasets across business types and risk appetites for testing.",
  },
];

export default function Landing() {
  return (
    <GridBackground>
      <header className="flex items-center justify-between px-6 py-6 sm:px-12">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-7 w-7 text-brand" />
          <div>
            <div className="text-lg font-bold leading-tight">FraudLab AI</div>
            <div className="text-[10px] uppercase tracking-widest text-slate-400">Risk Intelligence</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/login"
            className="rounded-lg px-4 py-2 text-sm font-semibold text-slate-200 transition hover:text-white"
          >
            Log in
          </Link>
          <Link
            to="/signup"
            className="rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-700"
          >
            Get Started
          </Link>
        </div>
      </header>

      <main className="flex flex-1 flex-col items-center px-6 pb-24 pt-12 text-center sm:pt-20">
        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-slate-300">
          <Gauge className="h-3.5 w-3.5 text-brand" /> PR-AUC 0.903 · Recall 0.819 on the champion model
        </span>

        <h1 className="mt-6 max-w-3xl text-4xl font-bold leading-tight sm:text-6xl">
          Catch fraud before it
          <span className="text-brand"> costs you.</span>
        </h1>

        <p className="mt-5 max-w-xl text-base text-slate-400 sm:text-lg">
          FraudLab AI scores transactions in real time, explains every decision with SHAP, and
          drafts investigation reports so your team can act with confidence.
        </p>

        <div className="mt-8 flex flex-col items-center gap-3 sm:flex-row">
          <Link
            to="/signup"
            className="inline-flex items-center gap-2 rounded-xl bg-brand px-6 py-3 text-sm font-semibold text-white transition hover:bg-brand-700"
          >
            Create free account <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-6 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
          >
            Log in to dashboard
          </Link>
        </div>

        <div className="mt-20 grid w-full max-w-5xl grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-2xl border border-white/10 bg-white/5 p-6 text-left backdrop-blur-sm transition hover:border-brand/40 hover:bg-white/10"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand/15">
                <Icon className="h-5 w-5 text-brand" />
              </div>
              <h3 className="mt-4 text-sm font-bold text-white">{title}</h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-400">{description}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="border-t border-white/10 px-6 py-6 text-center text-xs text-slate-500">
        FraudLab AI · Risk Intelligence Platform
      </footer>
    </GridBackground>
  );
}
