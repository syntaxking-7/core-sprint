"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import AnalysisDashboard from "@/components/AnalysisDashboard";
import { Loader2 } from "lucide-react";

interface AnalysisPayload {
    report: string; // JSON string
    created_at: string;
    ticker: string;
  id: string;
}

export default function AnalysisPage() {
  const { id } = useParams();
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const res = await fetch(`/api/analyses/${id}`);
        if (!res.ok) throw new Error("Failed to fetch analysis");
        const data = await res.json();
        setAnalysis(data);
      } catch (err) {
        console.error("Error fetching analysis:", err);
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchAnalysis();
  }, [id]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <Loader2 className="animate-spin h-8 w-8 text-gray-500" />
        <span className="ml-2 text-gray-600">Loading analysis...</span>
      </div>
    );
  }

  if (!analysis) {
    return (
      <p className="text-center text-gray-600 mt-10">
        No analysis found for this ID.
      </p>
    );
  }
  console.log(analysis)
  return <AnalysisDashboard payload={analysis} />;
}
