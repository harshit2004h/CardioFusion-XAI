"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { Activity, ArrowRight, ArrowUpRight, Check, ChevronRight, CircleAlert, FileHeart, FlaskConical, HeartPulse, History, LockKeyhole, Settings, Sparkles, UploadCloud, Video, X } from "lucide-react";
import { SignInButton, UserButton, useAuth, useUser } from "@clerk/nextjs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ModalityKey = "biomarkers" | "ecg" | "echo";
type Finding = { rule?: string; threshold?: number; threshold_assessment?: string; present?: boolean | null; point_estimate_flag?: boolean };
type Measurement = { name?: string; unit?: string | null; value?: number; interval?: { lower?: number; upper?: number; nominal_coverage?: number | null } };
type Risk = {
  disease?: string;
  display_name?: string;
  risk?: number | null;
  model_score?: number | null;
  decision_threshold?: number;
  component_scores?: Record<string, number>;
  uncalibrated_components?: string[];
  status?: string;
  message?: string;
  confirmation_message?: string;
  recommendations?: string[];
  calibration_status?: string;
  primary_modality?: string;
  detected_from?: string[];
  measurement?: Measurement;
  finding?: Finding;
};
type ModalityResult = { status?: string; available?: boolean; details?: { error?: string; missing_required_features?: string[]; model?: string } };
type Report = {
  id: string;
  label: string;
  status: string;
  modalities: string[];
  result?: { disease_results?: Risk[]; modalities?: Record<string, ModalityResult> };
  recommendations?: { source?: string; model?: string; summary?: string; recommendations?: string[] };
  errorMessage?: string | null;
  createdAt: string;
};
type Profile = { phone: string; address: string; username: string };
type Notice = { text: string; tone: "ok" | "error" };

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const apiUrl = process.env.NEXT_PUBLIC_SERVER_URL ?? "http://localhost:4000";
let reportsCache: { userId: string; reports: Report[] } | null = null;

// Escapes keep these characters intact regardless of how the file is re-saved.
const DOT = "·";
const DASH = "–";

const MODALITIES: Array<{ key: ModalityKey; label: string; source: string; Icon: LucideIcon }> = [
  { key: "ecg", label: "ECG", source: "From your ECG", Icon: Activity },
  { key: "echo", label: "Echo video", source: "From your echo", Icon: Video },
  { key: "biomarkers", label: "Blood report", source: "From your blood report", Icon: FlaskConical },
];
const modalityLabel = (key: string) => MODALITIES.find((item) => item.key === key)?.label ?? key;

// What each other evidence type adds to a result. Wording stays honest about what
// the connected models can and cannot do: context is never presented as confirmation.
const NEXT_EVIDENCE: Record<string, Array<{ modality: ModalityKey; why: string }>> = {
  acute_mi: [
    { modality: "ecg", why: "Adds an independent ECG myocardial-infarction label signal (a separate endpoint, shown side by side)." },
    { modality: "echo", why: "Adds a measured ejection fraction for pumping-function context." },
  ],
  st_t_abnormalities: [
    { modality: "biomarkers", why: "Adds a blood-based cardiovascular risk score (Framingham 10-year CHD) as context." },
    { modality: "echo", why: "Adds a measured ejection fraction for pumping-function context." },
  ],
  atrial_fibrillation: [
    { modality: "echo", why: "Adds a measured ejection fraction; AF and reduced pumping function are often assessed together." },
    { modality: "biomarkers", why: "Adds blood-based risk context. Blood values cannot confirm AF." },
  ],
  bradyarrhythmia: [
    { modality: "biomarkers", why: "Adds blood-based risk context. Blood values cannot confirm a rhythm finding." },
    { modality: "echo", why: "Adds a measured ejection fraction for pumping-function context." },
  ],
  lvh_rvh: [
    { modality: "echo", why: "Adds a measured ejection fraction. This echo model does not measure wall thickness, so it cannot confirm hypertrophy." },
    { modality: "biomarkers", why: "Adds blood-based risk context such as blood pressure and cholesterol." },
  ],
  myocardial_ischemia: [{ modality: "biomarkers", why: "Adds blood-based cardiovascular risk context." }],
  ejection_fraction: [
    { modality: "biomarkers", why: "Adds a heart-failure-cohort blood score (creatinine, sodium, and related values) as context." },
    { modality: "ecg", why: "Adds rhythm, conduction, and hypertrophy label signals." },
  ],
  hf_mortality: [
    { modality: "echo", why: "Adds a measured ejection fraction." },
    { modality: "ecg", why: "Adds rhythm and conduction label signals." },
  ],
  ten_year_chd_risk: [
    { modality: "ecg", why: "Adds ST/T, rhythm, and myocardial-infarction label signals." },
    { modality: "echo", why: "Adds a measured ejection fraction." },
  ],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function apiFetch(path: string, token: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${apiUrl.replace(/\/+$/, "")}${path}`, { ...options, headers });
}

async function responseError(response: Response) {
  const payload = await response.json().catch(() => null) as { error?: unknown } | null;
  return typeof payload?.error === "string" ? payload.error : `Request failed (${response.status}).`;
}

function isProbability(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1; }
function isFiniteNumber(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value); }
function hasCalibratedRisk(item: Risk) { return isProbability(item.risk); }
function componentScores(item: Risk) { return Object.entries(item.component_scores ?? {}).filter(([, value]) => isProbability(value)); }
function hasMeasurement(item: Risk) { return isFiniteNumber(item.measurement?.value); }
function hasScore(item: Risk) { return hasCalibratedRisk(item) || isProbability(item.model_score) || componentScores(item).length > 0; }
// A result is shown only when an uploaded, processed modality produced a value for it.
function hasModelOutput(item: Risk) { return hasScore(item) || hasMeasurement(item); }
function formatPercent(value: number) { return `${Math.round(value * 100)}%`; }
function formatDate(value: string, long = true) { return new Date(value).toLocaleDateString("en", long ? { month: "long", day: "numeric", year: "numeric" } : { month: "short", day: "numeric", year: "numeric" }); }
function formatMeasurement(value: number, unit?: string | null) { return `${value.toFixed(1)}${unit === "%" ? "%" : unit ? ` ${unit}` : ""}`; }
function resultKey(item: Risk) { return item.disease ?? item.display_name ?? ""; }
function analysisHref(report: Report) { return `/dashboard/analysis?report=${encodeURIComponent(report.id)}`; }

function modalityOf(item: Risk): ModalityKey | undefined {
  const raw = item.primary_modality
    ?? item.detected_from?.[0]
    ?? item.recommendations?.map((text) => /^Upload (\w+)\.$/.exec(text)?.[1]).find(Boolean);
  return raw === "biomarkers" || raw === "ecg" || raw === "echo" ? raw : undefined;
}

function analyzedModalities(report?: Report) {
  const analyzed = new Set<ModalityKey>();
  for (const { key } of MODALITIES) if (report?.result?.modalities?.[key]?.status === "processed") analyzed.add(key);
  return analyzed;
}

function modalityState(report: Report, key: ModalityKey): { state: "ok" | "missing" | "issue"; note: string } {
  const detail = report.result?.modalities?.[key];
  if (!detail) return report.modalities.includes(key) ? { state: "issue", note: "Not analyzed" } : { state: "missing", note: "Not uploaded" };
  if (detail.status === "processed") return { state: "ok", note: "Analyzed" };
  if (detail.status === "not_provided") return { state: "missing", note: "Not uploaded" };
  return { state: "issue", note: detail.status === "partial_extraction" ? "Missing fields" : "Couldn’t analyze" };
}

function scoredResults(report?: Report) { return (report?.result?.disease_results ?? []).filter(hasModelOutput); }
function topCalibrated(report?: Report, count = 4) {
  return (report?.result?.disease_results ?? []).filter(hasCalibratedRisk).sort((a, b) => (b.risk as number) - (a.risk as number)).slice(0, count);
}

type MeasurementGroup = { key: string; title: string; modality?: ModalityKey; measurement: Measurement & { value: number }; items: Risk[] };
// Several registry rules can read the same measured value (EF vs. 40% and 50%).
// Show that value once, with each rule as a line underneath, rather than repeating it.
function groupMeasurements(results: Risk[]): MeasurementGroup[] {
  const groups = new Map<string, MeasurementGroup>();
  for (const item of results) {
    if (!hasMeasurement(item) || hasScore(item)) continue;
    const measurement = item.measurement as Measurement & { value: number };
    const key = `${measurement.name ?? "measurement"}:${measurement.value}`;
    const group = groups.get(key) ?? {
      key,
      title: measurement.name === "EF" ? "Left Ventricular Ejection Fraction" : item.display_name ?? "Measurement",
      modality: modalityOf(item),
      measurement,
      items: [],
    };
    group.items.push(item);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function findingVerdict(finding: Finding) {
  const threshold = isFiniteNumber(finding.threshold) ? `${finding.threshold}%` : "the threshold";
  switch (finding.threshold_assessment) {
    case "interval_below_threshold": return { tone: "flag", label: "Below", text: `The whole interval is below ${threshold}.` };
    case "interval_at_or_above_threshold": return { tone: "clear", label: "Not below", text: `The whole interval is at or above ${threshold}.` };
    case "interval_crosses_threshold": return { tone: "unsure", label: "Indeterminate", text: `The interval spans ${threshold}, so this rule can’t be settled from this read.` };
    default: return finding.point_estimate_flag || finding.present
      ? { tone: "flag", label: "Below", text: `The measured value is below ${threshold} (no interval available).` }
      : { tone: "clear", label: "Not below", text: `The measured value is at or above ${threshold} (no interval available).` };
  }
}

function Brand() { return <Link className="brand" href="/"><span className="brand-mark"><Activity size={18} /></span><span>cardio<span>fusion</span></span></Link>; }
function displayName(user: ReturnType<typeof useUser>["user"]) { if (!user) return "Your workspace"; const name = [user.firstName, user.lastName].filter(Boolean).join(" "); return name || user.fullName || user.username || user.emailAddresses[0]?.emailAddress?.split("@")[0] || "Your workspace"; }

const navItems = [
  ["/dashboard", "Overview", Activity],
  ["/dashboard/upload", "New assessment", UploadCloud],
  ["/dashboard/history", "History", History],
  ["/dashboard/analysis", "Analysis", HeartPulse],
  ["/dashboard/settings", "Settings", Settings],
] as const;

// ---------------------------------------------------------------------------
// Workspace shell
// ---------------------------------------------------------------------------

export function Workspace() {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const { user } = useUser();
  const pathname = usePathname();
  const router = useRouter();
  const [reports, setReports] = useState<Report[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile>({ phone: "", address: "", username: "" });
  const [files, setFiles] = useState<Record<ModalityKey, File[]>>({ biomarkers: [], ecg: [], echo: [] });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const name = displayName(user);
  const activePath = pathname === "/" ? "/dashboard" : pathname;
  const latest = reports[0];
  const selected = reports.find((report) => report.id === selectedReportId) ?? latest;

  // Track one condition's calibrated value across reports; reports without that value are skipped, never filled in.
  const trend = useMemo(() => {
    const tracked = topCalibrated(selected, 1)[0];
    if (!tracked) return null;
    const key = resultKey(tracked);
    const points = reports.slice(0, 8).reverse().flatMap((report) => {
      const value = report.result?.disease_results?.find((item) => resultKey(item) === key && hasCalibratedRisk(item))?.risk;
      return isProbability(value) ? [{ value, date: report.createdAt, id: report.id }] : [];
    });
    return { name: tracked.display_name ?? key, points };
  }, [reports, selected]);

  async function load(force = false) {
    try {
      const token = await getToken();
      if (!token || !user) return;
      const cached = !force && reportsCache?.userId === user.id;
      if (cached) { setReports(reportsCache!.reports); setLoaded(true); }
      const reportRequest = cached ? null : apiFetch("/api/reports", token);
      const profileRequest = pathname === "/dashboard/settings" ? apiFetch("/api/profile", token) : null;
      const [reportResponse, profileResponse] = await Promise.all([reportRequest, profileRequest]);
      if (reportResponse && !reportResponse.ok) throw new Error(await responseError(reportResponse));
      if (profileResponse && !profileResponse.ok) throw new Error(await responseError(profileResponse));
      if (reportResponse) {
        const data = await reportResponse.json() as { reports: Report[] };
        reportsCache = { userId: user.id, reports: data.reports };
        setReports(data.reports);
      }
      if (profileResponse) {
        const data = await profileResponse.json();
        setProfile({ phone: data.user.phone ?? "", address: data.user.address ?? "", username: data.user.username ?? "" });
      }
    } catch (error) {
      setNotice({ text: error instanceof Error ? error.message : "Could not load your workspace.", tone: "error" });
    } finally {
      setLoaded(true);
    }
  }
  // Restore a report selected from history or a link when the page remounts.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setSelectedReportId(new URLSearchParams(window.location.search).get("report")); }, [pathname]);
  // Hydrate the authenticated workspace from the API after Clerk resolves.
  // eslint-disable-next-line react-hooks/set-state-in-effect, react-hooks/exhaustive-deps
  useEffect(() => { if (isSignedIn) void load(); }, [isSignedIn, user]);

  async function submitReport() {
    const chosen = (Object.entries(files) as Array<[ModalityKey, File[]]>).filter(([, picked]) => picked.length);
    if (!chosen.length) { setNotice({ text: "Choose at least one file first.", tone: "error" }); return; }
    const ecgFiles = files.ecg;
    const ecgExtensions = ecgFiles.map((file) => file.name.split(".").pop()?.toLowerCase());
    const hasWfdbFile = ecgExtensions.some((extension) => extension === "dat" || extension === "hea");
    if (hasWfdbFile && !(ecgFiles.length === 2 && ecgExtensions.includes("dat") && ecgExtensions.includes("hea"))) {
      setNotice({ text: "A WFDB ECG needs one matching .dat and .hea file.", tone: "error" }); return;
    }
    if (hasWfdbFile && ecgFiles.length === 2) {
      const stems = ecgFiles.map((file) => file.name.replace(/\.(dat|hea)$/i, "").toLowerCase());
      if (stems[0] !== stems[1]) { setNotice({ text: "The WFDB .dat and .hea files must have the same base filename.", tone: "error" }); return; }
    }
    if (!hasWfdbFile && ecgFiles.length > 1) {
      setNotice({ text: "Upload one ECG file, or one matching .dat and .hea pair.", tone: "error" }); return;
    }
    setBusy(true);
    setNotice({ text: "Uploading and reading your evidence. This can take a minute…", tone: "ok" });
    try {
      const token = await getToken();
      if (!token) throw new Error("Your session expired. Sign in again and retry.");
      const form = new FormData();
      chosen.forEach(([key, picked]) => picked.forEach((file) => form.append(key, file)));
      form.append("label", "Cardiovascular assessment");
      const response = await apiFetch("/api/reports", token, { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json() as { report: Report };
      reportsCache = null;
      setFiles({ biomarkers: [], ecg: [], echo: [] });
      setNotice({
        text: data.report.status === "PARTIAL"
          ? "Assessment saved with partial results. Some evidence couldn’t be analyzed; details are below."
          : "Assessment complete.",
        tone: "ok",
      });
      await load(true);
      router.push(analysisHref(data.report));
    } catch (error) {
      setNotice({ text: error instanceof Error ? error.message : "We could not complete this assessment.", tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function saveProfile(event: React.FormEvent) {
    event.preventDefault();
    try {
      const token = await getToken();
      if (!token) throw new Error("Your session expired. Sign in again and retry.");
      const response = await apiFetch("/api/profile", token, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(profile),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setNotice({ text: "Settings saved.", tone: "ok" });
    } catch (error) {
      setNotice({ text: error instanceof Error ? error.message : "Could not save your settings.", tone: "error" });
    }
  }

  if (!isLoaded) return <div className="loading-screen">{"Preparing your private workspace…"}</div>;
  if (!isSignedIn) {
    return <div className="signed-out">
      <Brand />
      <h1 className="serif">Sign in to open your workspace.</h1>
      <p>Your assessments are private to your account.</p>
      <div className="signed-out-actions">
        <SignInButton mode="modal"><button className="coral-button">Sign in <ArrowUpRight size={16} /></button></SignInButton>
        <Link className="text-link" href="/">Back to home</Link>
      </div>
    </div>;
  }

  const section = activePath === "/dashboard/upload" ? "upload" : activePath === "/dashboard/history" ? "history" : activePath === "/dashboard/settings" ? "settings" : activePath === "/dashboard/analysis" ? "analysis" : "overview";
  const heading = { overview: `Good to see you, ${name}.`, upload: "Start an assessment.", history: "Your history.", analysis: "Analysis.", settings: "Settings." }[section];
  const eyebrow = { overview: "Your evidence desk", upload: "Add evidence", history: "A record over time", analysis: "A closer look", settings: "Your details" }[section];

  return <main className="workspace">
    <aside className="sidebar">
      <Brand />
      <div className="sidebar-user"><div className="avatar">{name[0]?.toUpperCase()}</div><div><b>{name}</b><small>Private account</small></div></div>
      <nav className="side-nav">{navItems.map(([href, label, Icon]) => <Link className={activePath === href ? "active" : ""} href={href} key={href} title={label}><Icon size={17} /> {label}{label === "History" && reports.length ? <span>{reports.length}</span> : null}</Link>)}</nav>
      <div className="sidebar-bottom"><div className="privacy-note"><LockKeyhole size={15} /><span>Your data stays yours.<br /><small>Private by default.</small></span></div><UserButton afterSignOutUrl="/" /></div>
    </aside>
    <section className="workspace-main">
      <header className="workspace-header"><div><div className="eyebrow">{eyebrow}</div><h1 className="serif">{heading}</h1></div></header>
      {notice && <div className={`toast ${notice.tone}`} role={notice.tone === "error" ? "alert" : "status"}>{notice.tone === "error" ? <CircleAlert size={15} /> : <Check size={15} />}<span>{notice.text}</span><button onClick={() => setNotice(null)} aria-label="Dismiss"><X size={14} /></button></div>}
      {section === "overview" && <Overview report={selected} isLatest={selected?.id === latest?.id} loaded={loaded} trend={trend} />}
      {section === "upload" && <UploadPanel files={files} setFiles={setFiles} busy={busy} submit={submitReport} />}
      {section === "history" && <HistoryPanel reports={reports} loaded={loaded} />}
      {section === "analysis" && <AnalysisPanel report={selected} loaded={loaded} />}
      {section === "settings" && <SettingsPanel profile={profile} setProfile={setProfile} save={saveProfile} />}
    </section>
  </main>;
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

function ModalityChips({ report, compact = false }: { report: Report; compact?: boolean }) {
  return <div className={`modality-strip ${compact ? "compact" : ""}`}>{MODALITIES.map(({ key, label, Icon }) => {
    const { state, note } = modalityState(report, key);
    const detail = report.result?.modalities?.[key]?.details;
    const issue = detail?.error ?? (detail?.missing_required_features?.length ? `Missing: ${detail.missing_required_features.join(", ")}` : undefined);
    const body = <><Icon size={15} /><span><b>{label}</b><small>{note}</small></span>{state === "missing" ? <ArrowRight size={14} /> : state === "ok" ? <Check size={14} /> : <CircleAlert size={14} />}</>;
    return state === "missing"
      ? <Link key={key} className="modality-chip missing" href="/dashboard/upload" title={`Add ${label.toLowerCase()}`}>{body}</Link>
      : <span key={key} className={`modality-chip ${state}`} title={issue}>{body}</span>;
  })}</div>;
}

function Overview({ report, isLatest, loaded, trend }: { report?: Report; isLatest: boolean; loaded: boolean; trend: { name: string; points: Array<{ value: number; date: string; id: string }> } | null }) {
  if (!report) {
    return <div className="dashboard-content fade-up"><div className="empty-state">
      <HeartPulse size={30} />
      <h2 className="serif">{loaded ? "Your first read is waiting." : "Loading your assessments…"}</h2>
      <p>Bring whatever evidence you have: a blood report, an ECG, or an echo video.</p>
      {loaded ? <Link className="coral-button" href="/dashboard/upload"><UploadCloud size={16} /> Start an assessment</Link> : null}
    </div></div>;
  }
  const top = topCalibrated(report, 5);
  const ef = groupMeasurements(scoredResults(report))[0];
  const highlights: Array<{ key: string; label: string; value: string; note: string }> = [
    ...(ef ? [{ key: ef.key, label: ef.title, value: formatMeasurement(ef.measurement.value, ef.measurement.unit), note: "Measured from echo" }] : []),
    ...top.map((item) => ({ key: resultKey(item), label: item.display_name ?? resultKey(item), value: formatPercent(item.risk as number), note: "Calibrated probability" })),
  ].slice(0, 2);
  const guidance = report.recommendations;

  return <div className="dashboard-content fade-up">
    <div className="overview-top">
      <div><span className="eyebrow">{isLatest ? "Latest assessment" : "Selected assessment"}</span><h2 className="serif">{report.label}</h2><p>{formatDate(report.createdAt)}</p></div>
      <div className="overview-actions">
        <Link className="ghost-button" href={analysisHref(report)}>View full analysis <ArrowRight size={16} /></Link>
        <Link className="coral-button" href="/dashboard/upload"><UploadCloud size={16} /> Add evidence</Link>
      </div>
    </div>
    <div className="metric-grid">
      <div className="metric-card wide"><div className="metric-head"><span>Evidence in this read</span><Sparkles size={17} /></div><ModalityChips report={report} compact /></div>
      {highlights.map((item) => <Link className="metric-card link-card" href={analysisHref(report)} key={item.key}><span>{item.label}</span><strong>{item.value}</strong><small>{item.note}</small></Link>)}
    </div>
    <div className="detail-grid">
      <div className="results-panel">
        <div className="panel-heading"><div><span className="eyebrow">Strongest signals</span><h3 className="serif">What the models noticed</h3></div><HeartPulse size={21} /></div>
        {top.length ? <div className="signal-list">{top.map((item) => <Link className="signal-row" href={analysisHref(report)} key={resultKey(item)}>
          <span>{item.display_name}</span><b>{formatPercent(item.risk as number)}</b>
          <i><em style={{ width: `${(item.risk as number) * 100}%` }} /></i>
        </Link>)}</div> : <p className="empty-copy">No calibrated probability came from this evidence.{ef ? " The echo measurement is shown above." : ""}</p>}
      </div>
      <div className="context-panel">
        <div className="panel-heading"><div><span className="eyebrow">Trajectory</span><h3 className="serif">Across your reads</h3></div><Activity size={20} /></div>
        <TrendChart trend={trend} />
      </div>
    </div>
    {guidance?.summary || guidance?.recommendations?.length ? <div className="guidance-strip">
      <Sparkles size={18} />
      <div><span className="eyebrow">Suggested next steps</span><p>{guidance.summary ?? guidance.recommendations?.[0]}</p></div>
      <Link className="text-link" href={analysisHref(report)}>Read more <ChevronRight size={16} /></Link>
    </div> : null}
  </div>;
}

function TrendChart({ trend }: { trend: { name: string; points: Array<{ value: number; date: string; id: string }> } | null }) {
  if (!trend) return <p className="empty-copy chart-empty">A trend appears once an assessment includes a calibrated probability.</p>;
  if (trend.points.length < 2) return <p className="empty-copy chart-empty">Tracking <b>{trend.name}</b>. A second assessment with this signal will draw the trend.</p>;
  const step = 84 / (trend.points.length - 1);
  const coords = trend.points.map((point, index) => ({ ...point, x: 8 + index * step, y: 76 - point.value * 66 }));
  return <div className="trend">
    <small>{trend.name}</small>
    <svg className="risk-chart" viewBox="0 0 100 86" role="img" aria-label={`${trend.name} across ${trend.points.length} assessments`}>
      <line x1="8" x2="92" y1="76" y2="76" className="chart-axis" />
      <line x1="8" x2="92" y1="43" y2="43" className="chart-mid" />
      <polyline points={coords.map((point) => `${point.x},${point.y}`).join(" ")} />
      {coords.map((point) => <circle key={point.id} cx={point.x} cy={point.y} r="1.6"><title>{`${formatDate(point.date, false)}: ${formatPercent(point.value)}`}</title></circle>)}
    </svg>
    <div className="chart-dates"><span>{formatDate(trend.points[0].date, false)}</span><span>{formatDate(trend.points[trend.points.length - 1].date, false)}</span></div>
  </div>;
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

function UploadPanel({ files, setFiles, busy, submit }: { files: Record<ModalityKey, File[]>; setFiles: React.Dispatch<React.SetStateAction<Record<ModalityKey, File[]>>>; busy: boolean; submit: () => void }) {
  const inputs: Array<{ key: ModalityKey; label: string; hint: string; accept: string; unlocks: string; Icon: LucideIcon }> = [
    { key: "biomarkers", label: "Blood report", hint: "PDF, JPG, JPEG or PNG", accept: ".pdf,.jpg,.jpeg,.png", unlocks: "Heart-attack label score, 10-year CHD risk, heart-failure cohort score", Icon: FlaskConical },
    { key: "ecg", label: "ECG", hint: "DICOM, DAT + HEA, CSV, SCP or EDF", accept: ".dcm,.dicom,.dat,.hea,.csv,.scp,.edf", unlocks: "Rhythm, ST/T, hypertrophy and infarction label signals", Icon: Activity },
    { key: "echo", label: "Echo video", hint: "AVI or MP4", accept: ".avi,.mp4", unlocks: "Measured ejection fraction with an uncertainty interval", Icon: Video },
  ];
  const chosenCount = Object.values(files).filter((picked) => picked.length).length;
  return <div className="dashboard-content fade-up">
    <div className="upload-intro"><p>Upload one or more pieces of evidence. Anything you leave out stays visibly missing instead of becoming an assumption.</p><span><LockKeyhole size={15} /> Encrypted in transit</span></div>
    <div className="upload-grid">{inputs.map(({ key, label, hint, accept, unlocks, Icon }) => {
      const picked = files[key];
      return <div className="upload-tile-wrap" key={key}>
        <label className={`upload-tile ${picked.length ? "chosen" : ""}`}>
          <Icon size={21} /><b>{label}</b>
          <small>{picked.length ? picked.map((file) => file.name).join(", ") : hint}</small>
          <em>{unlocks}</em>
          {picked.length ? <Check className="upload-check" size={18} /> : <UploadCloud className="upload-arrow" size={18} />}
          <input type="file" accept={accept} multiple={key === "ecg"} disabled={busy} onChange={(event) => {
            const fileInput = event.currentTarget;
            const newlySelected = Array.from(fileInput.files ?? []);
            setFiles((current) => {
              const combined = key === "ecg" ? [...current.ecg, ...newlySelected] : newlySelected;
              const unique = Array.from(new Map(combined.map((file) => [`${file.name}:${file.size}:${file.lastModified}`, file])).values());
              return { ...current, [key]: unique };
            });
            fileInput.value = "";
          }} />
        </label>
        {picked.length ? <button type="button" className="upload-clear" disabled={busy} onClick={() => setFiles((current) => ({ ...current, [key]: [] }))}>Clear selection</button> : null}
      </div>;
    })}</div>
    <div className="upload-footer"><span>For a WFDB ECG, select both the matching .dat and .hea files, together or one at a time.</span><button className="coral-button" onClick={submit} disabled={busy || !chosenCount}>{busy ? "Reading…" : `Begin assessment${chosenCount ? ` (${chosenCount})` : ""}`} <ArrowUpRight size={16} /></button></div>
  </div>;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function HistoryPanel({ reports, loaded }: { reports: Report[]; loaded: boolean }) {
  if (!reports.length) {
    return <div className="dashboard-content fade-up"><div className="empty-state">
      <History size={30} />
      <h2 className="serif">{loaded ? "Your story starts here." : "Loading your history…"}</h2>
      <p>Completed assessments will appear in this timeline.</p>
      {loaded ? <Link className="coral-button" href="/dashboard/upload"><UploadCloud size={16} /> Start an assessment</Link> : null}
    </div></div>;
  }
  return <div className="dashboard-content fade-up"><div className="history-list">{reports.map((report) => {
    const signals = scoredResults(report).length;
    return <Link className="history-row" key={report.id} href={analysisHref(report)}>
      <div className="report-icon"><FileHeart size={18} /></div>
      <div className="history-main"><b>{report.label}</b><span>{formatDate(report.createdAt, false)} {DOT} {report.modalities.map(modalityLabel).join(` ${DOT} `) || "No evidence"}{report.status !== "FAILED" ? ` ${DOT} ${signals} result${signals === 1 ? "" : "s"}` : ""}</span></div>
      <span className={`history-status ${report.status.toLowerCase()}`}>{report.status.toLowerCase()}</span>
      <ChevronRight size={17} />
    </Link>;
  })}</div></div>;
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

function NextEvidence({ diseaseKey, analyzed }: { diseaseKey: string; analyzed: Set<ModalityKey> }) {
  const all = NEXT_EVIDENCE[diseaseKey] ?? [];
  const missing = all.filter((option) => !analyzed.has(option.modality));
  if (!all.length) return <p className="next-evidence-done">No other evidence type in this service adds to this result.</p>;
  if (!missing.length) return <p className="next-evidence-done"><Check size={13} /> All evidence this result can use is included.</p>;
  return <div className="next-evidence">
    <span>For a fuller picture</span>
    {missing.map((option) => <Link key={option.modality} href="/dashboard/upload"><b>Add {modalityLabel(option.modality).toLowerCase()}</b><small>{option.why}</small><ArrowRight size={14} /></Link>)}
  </div>;
}

function SignalCard({ item, analyzed }: { item: Risk; analyzed: Set<ModalityKey> }) {
  const calibrated = hasCalibratedRisk(item);
  const components = componentScores(item);
  const value = calibrated ? item.risk as number : isProbability(item.model_score) ? item.model_score : undefined;
  const threshold = isProbability(item.decision_threshold) ? item.decision_threshold : undefined;
  const badge = components.length && value === undefined
    ? { text: "Per-label scores", tone: "neutral" }
    : !calibrated
      ? { text: "Uncalibrated score", tone: "warn" }
      : threshold !== undefined
        ? (value as number) >= threshold ? { text: `At or above ${formatPercent(threshold)}`, tone: "high" } : { text: `Below ${formatPercent(threshold)}`, tone: "low" }
        : { text: "Calibrated", tone: "neutral" };
  const explanation = item.message ?? item.confirmation_message ?? (calibrated
    ? "The model’s calibrated probability for this label on your recording. A research signal, not a diagnosis."
    : "A raw model score without calibration. It is not a patient risk and should not be read as one.");

  return <article className="result-card">
    <div className="result-card-top"><h4>{item.display_name ?? "Unnamed condition"}</h4><span className={`pill ${badge.tone}`}>{badge.text}</span></div>
    {value !== undefined ? <>
      <div className="result-value"><strong>{formatPercent(value)}</strong><small>{calibrated ? "Calibrated probability" : "Model score"}</small></div>
      <div className={`prob-track ${calibrated ? "" : "uncalibrated"}`} role="img" aria-label={`${formatPercent(value)}${threshold !== undefined ? `, decision threshold ${formatPercent(threshold)}` : ""}`}>
        <i style={{ width: `${value * 100}%` }} />
        {threshold !== undefined ? <b style={{ left: `${threshold * 100}%` }} title={`Decision threshold ${formatPercent(threshold)}`} /> : null}
      </div>
    </> : null}
    {components.length ? <div className="component-list">{components.map(([label, score]) => <div key={label}><span>{label}</span><small>{item.uncalibrated_components?.includes(label) ? "uncalibrated" : "calibrated"}</small><b>{formatPercent(score)}</b></div>)}</div> : null}
    <p className="result-explanation">{explanation}</p>
    <NextEvidence diseaseKey={resultKey(item)} analyzed={analyzed} />
  </article>;
}

function MeasurementCard({ group, analyzed }: { group: MeasurementGroup; analyzed: Set<ModalityKey> }) {
  const { measurement } = group;
  const interval = measurement.interval;
  const bounds = isFiniteNumber(interval?.lower) && isFiniteNumber(interval?.upper) && interval.upper >= interval.lower ? { lower: interval.lower, upper: interval.upper } : undefined;
  const percentScale = measurement.unit === "%" && measurement.value >= 0 && measurement.value <= 100;
  const findings = group.items.filter((item) => item.finding).sort((a, b) => (b.finding?.threshold ?? 0) - (a.finding?.threshold ?? 0));
  const coverage = isFiniteNumber(interval?.nominal_coverage) ? `${Math.round(interval.nominal_coverage * 100)}% interval` : "Interval";
  const nextKey = measurement.name === "EF" ? "ejection_fraction" : resultKey(group.items[0]);

  return <article className="result-card measurement-card">
    <div className="result-card-top"><h4>{group.title}</h4><span className="pill neutral">Measured {DOT} not a risk</span></div>
    <div className="result-value"><strong className="measure">{formatMeasurement(measurement.value, measurement.unit)}</strong>
      <small>{bounds ? `${coverage}: ${formatMeasurement(bounds.lower, measurement.unit)} ${DASH} ${formatMeasurement(bounds.upper, measurement.unit)}` : "No uncertainty interval available"}</small></div>
    {percentScale ? <div className="measure-scale">
      <div className="measure-track" role="img" aria-label={`${formatMeasurement(measurement.value, measurement.unit)}${bounds ? `, interval ${bounds.lower.toFixed(1)} to ${bounds.upper.toFixed(1)}` : ""}`}>
        {bounds ? <i style={{ left: `${Math.max(0, bounds.lower)}%`, width: `${Math.min(100, bounds.upper) - Math.max(0, bounds.lower)}%` }} /> : null}
        {findings.map((item) => isFiniteNumber(item.finding?.threshold) ? <s key={resultKey(item)} style={{ left: `${item.finding.threshold}%` }}><em>{item.finding.threshold}%</em></s> : null)}
        <b style={{ left: `${measurement.value}%` }} />
      </div>
      <div className="measure-axis"><span>0%</span><span>100%</span></div>
    </div> : null}
    {findings.length ? <div className="finding-list">
      <span>Threshold checks</span>
      {findings.map((item) => {
        const finding = item.finding as Finding;
        const verdict = findingVerdict(finding);
        return <div key={resultKey(item)}><b>{measurement.name ?? "Value"} below {finding.threshold}%?</b><em className={`verdict ${verdict.tone}`}>{verdict.label}</em><small>{verdict.text}</small></div>;
      })}
      <p>Threshold checks are flags on a measurement, not diagnoses.</p>
    </div> : null}
    <NextEvidence diseaseKey={nextKey} analyzed={analyzed} />
  </article>;
}

function AnalysisPanel({ report, loaded }: { report?: Report; loaded: boolean }) {
  if (!report) {
    return <div className="dashboard-content fade-up"><div className="analysis-empty">
      <HeartPulse size={30} />
      <h2 className="serif">{loaded ? "No assessment selected yet." : "Loading your assessments…"}</h2>
      <p>Start an assessment, or open one from your history.</p>
      {loaded ? <div className="empty-actions"><Link className="coral-button" href="/dashboard/upload"><UploadCloud size={16} /> Start an assessment</Link><Link className="ghost-button" href="/dashboard/history">Open history</Link></div> : null}
    </div></div>;
  }

  const analyzed = analyzedModalities(report);
  const results = scoredResults(report);
  const measurements = groupMeasurements(results);
  const signals = results.filter(hasScore);
  const sections = MODALITIES.map((modality) => ({
    ...modality,
    measurements: measurements.filter((group) => group.modality === modality.key),
    signals: signals.filter((item) => modalityOf(item) === modality.key).sort((a, b) => ((b.risk ?? b.model_score ?? 0) - (a.risk ?? a.model_score ?? 0))),
  })).filter((group) => group.measurements.length || group.signals.length);
  const unassigned = signals.filter((item) => !modalityOf(item));

  // Conditions that would be assessed if a missing modality were uploaded.
  const unlocks = MODALITIES.filter(({ key }) => !analyzed.has(key)).map((modality) => ({
    ...modality,
    names: (report.result?.disease_results ?? []).filter((item) => item.status === "primary_missing" && modalityOf(item) === modality.key).map((item) => item.display_name ?? resultKey(item)),
  })).filter((group) => group.names.length);

  const issues = Object.entries(report.result?.modalities ?? {}).filter(([, modality]) => modality.status && !["processed", "not_provided"].includes(modality.status));
  const guidance = report.recommendations;
  const failed = report.status === "FAILED";

  return <div className="dashboard-content fade-up analysis-content">
    <section className="analysis-hero">
      <div className="analysis-hero-icon"><HeartPulse size={24} /></div>
      <div className="analysis-hero-copy">
        <span className="eyebrow">Evidence review {DOT} {formatDate(report.createdAt)}</span>
        <h2 className="serif">{report.label}</h2>
        <p>{failed
          ? report.errorMessage ?? "This assessment could not be processed. Check the files and try again."
          : "Only results produced from the evidence you uploaded are shown. Probabilities, uncalibrated scores, and measurements are labeled separately. This is research support, not a diagnosis."}</p>
      </div>
      <span className={`assessment-badge ${report.status.toLowerCase()}`}>{report.status.toLowerCase()}</span>
      <ModalityChips report={report} />
    </section>

    {issues.length ? <section className="modality-issues" aria-label="Evidence that couldn't be analyzed">
      {issues.map(([modality, detail]) => <div key={modality}><strong>{modalityLabel(modality)} wasn{"’"}t fully analyzed</strong><span>{detail.details?.error ?? (detail.details?.missing_required_features?.length ? `The report is missing required fields: ${detail.details.missing_required_features.join(", ")}.` : `Processing status: ${detail.status?.replaceAll("_", " ")}.`)}</span></div>)}
    </section> : null}

    {results.length && (guidance?.summary || guidance?.recommendations?.length) ? <section className="recommendations-panel" aria-label="Suggested next steps">
      <div className="panel-heading"><div><span className="eyebrow">{guidance.source === "gemini" ? "AI-written guidance" : "General guidance"}</span><h3 className="serif">Suggested next steps</h3></div><Sparkles size={20} /></div>
      {guidance.summary ? <p>{guidance.summary}</p> : null}
      {guidance.recommendations?.length ? <ul>{guidance.recommendations.map((text) => <li key={text}>{text}</li>)}</ul> : null}
      <small>{guidance.source === "gemini" ? "Written by Gemini from the results below. It cannot change any score." : "Gemini was unavailable for this report, so standard guidance is shown."}</small>
    </section> : null}

    {sections.map((group) => <section className="source-section" key={group.key}>
      <div className="source-head"><span className="source-icon"><group.Icon size={17} /></span><div><span className="eyebrow">{group.source}</span><h3 className="serif">{group.measurements.length && !group.signals.length ? "Measurements" : `${group.signals.length + group.measurements.length} result${group.signals.length + group.measurements.length === 1 ? "" : "s"}`}</h3></div></div>
      <div className="result-grid">
        {group.measurements.map((measurement) => <MeasurementCard key={measurement.key} group={measurement} analyzed={analyzed} />)}
        {group.signals.map((item) => <SignalCard key={resultKey(item)} item={item} analyzed={analyzed} />)}
      </div>
    </section>)}
    {unassigned.length ? <section className="source-section"><div className="result-grid">{unassigned.map((item) => <SignalCard key={resultKey(item)} item={item} analyzed={analyzed} />)}</div></section> : null}

    {!results.length && !failed ? <div className="analysis-empty"><HeartPulse size={27} /><h3>No results from this evidence</h3><p>The uploaded files didn{"’"}t produce a score or measurement for any supported condition. Check the notes above, or add other evidence.</p><Link className="coral-button" href="/dashboard/upload"><UploadCloud size={16} /> Add evidence</Link></div> : null}

    {unlocks.length && !failed ? <section className="unlock-section">
      <div className="source-head"><span className="source-icon muted"><UploadCloud size={17} /></span><div><span className="eyebrow">Not assessed yet</span><h3 className="serif">Add evidence to see more</h3></div></div>
      <div className="unlock-grid">{unlocks.map(({ key, Icon, names }) => <Link className="unlock-card" href="/dashboard/upload" key={key}>
        <span className="unlock-title"><Icon size={18} /><b>Add {key === "biomarkers" ? "a blood report" : key === "ecg" ? "an ECG" : "an echo video"}</b></span>
        <small>The models could then assess:</small>
        <ul>{names.map((itemName) => <li key={itemName}>{itemName}</li>)}</ul>
        <span className="unlock-cta">Upload <ArrowRight size={14} /></span>
      </Link>)}</div>
    </section> : null}
  </div>;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

function SettingsPanel({ profile, setProfile, save }: { profile: Profile; setProfile: React.Dispatch<React.SetStateAction<Profile>>; save: (event: React.FormEvent) => void }) {
  return <div className="dashboard-content fade-up"><form className="settings-form" onSubmit={save}>
    <div className="settings-section">
      <div><span className="eyebrow">Profile</span><h2 className="serif">A few details, when you are ready.</h2><p>Your Clerk account is the source of truth for sign-in and your name.</p></div>
      <div className="field-stack">
        <label>Username<input value={profile.username} onChange={(event) => setProfile({ ...profile, username: event.target.value })} placeholder="How should we call you?" /></label>
        <label>Phone number<input value={profile.phone} onChange={(event) => setProfile({ ...profile, phone: event.target.value })} placeholder="Optional" /></label>
        <label>Address<textarea value={profile.address} onChange={(event) => setProfile({ ...profile, address: event.target.value })} placeholder="Optional" rows={3} /></label>
      </div>
    </div>
    <button className="coral-button" type="submit">Save settings <Check size={16} /></button>
  </form></div>;
}
