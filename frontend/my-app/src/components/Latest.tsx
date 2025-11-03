"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import Link from "next/link";

interface CompanyName {
  name: string;
  ticker: string;
}

interface AnalysisItem {
  id: string;
  ticker: string;
  company: string | CompanyName;
  created_at: string | null;
  score: number | null;
  grade: string | null;
}

export default function RecentAnalyses() {
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        const res = await fetch("/api/analyses/latest");
        const data = await res.json();
        if (data.analyses) {
          setAnalyses(data.analyses);
        }
      } catch (err) {
        console.error("Error fetching latest analyses:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyses();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-6">
        <Loader2 className="animate-spin h-6 w-6 text-gray-500" />
        <span className="ml-2 text-gray-600">Loading recent analyses...</span>
      </div>
    );
  }

  if (analyses.length === 0) {
    return <p className="text-gray-600 text-center py-6">No analyses available.</p>;
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {analyses.map((item) => {
        const displayName =
          typeof item.company === "string"
            ? item.company
            : item.company?.name || "Unknown Company";

        return (
          <Card key={item.id} className="shadow hover:shadow-lg transition">
            <CardHeader>
              <CardTitle className="flex justify-between items-center">
                <span>
                  {displayName} <span className="text-gray-500">({item.ticker})</span>
                </span>
                <span className="text-xs text-gray-400">
                  {item.created_at ? new Date(item.created_at).toLocaleDateString() : ""}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold">
                Score: {item.score ? item.score.toFixed(1) : "N/A"}
              </p>
              <p className="text-gray-600">Grade: {item.grade || "N/A"}</p>

              <Link
                href={`/analysis/${item.id}`}
                className="inline-block mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
              >
                View Full Analysis
              </Link>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
