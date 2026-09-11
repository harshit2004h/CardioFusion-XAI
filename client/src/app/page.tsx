"use client";

import { Landing } from "./landing";
import { useAuth } from "@clerk/nextjs";

function AuthenticatedHome() {
  const { isLoaded, isSignedIn } = useAuth();
  if (!isLoaded) return <div className="loading-screen">Preparing your home...</div>;
  return <Landing authenticated={Boolean(isSignedIn)} />;
}

export default function Home() {
  return process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ? <AuthenticatedHome /> : <Landing />;
}
