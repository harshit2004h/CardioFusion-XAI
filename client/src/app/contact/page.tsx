import Link from "next/link";
import { Mail, MessageCircle } from "lucide-react";

export default function ContactPage() { return <main className="public-page"><Link href="/" className="public-back">CardioFusion</Link><MessageCircle size={42} /><div className="eyebrow">Contact</div><h1 className="serif">Bring us a careful question.</h1><p>For product questions and workspace support, email the CardioFusion team. For medical advice, please contact a qualified clinician directly.</p><a className="coral-button" href="mailto:hello@cardiofusion.example"><Mail size={16} /> hello@cardiofusion.example</a></main>; }
