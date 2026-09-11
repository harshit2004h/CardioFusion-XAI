import Link from "next/link";
import { HeartPulse } from "lucide-react";

export default function AboutPage() { return <main className="public-page"><Link href="/" className="public-back">CardioFusion</Link><HeartPulse size={42} /><div className="eyebrow">About CardioFusion</div><h1 className="serif">A considered view of cardiovascular evidence.</h1><p>CardioFusion keeps blood work, ECG signals, and echo video distinct, then brings them together so people can ask better questions with their care team.</p><Link href="/dashboard" className="coral-button">Open workspace</Link></main>; }
