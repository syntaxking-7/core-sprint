"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import AnalysisDashboard from "@/components/AnalysisDashboard";
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL
interface AnalysisResponse {
  report: string;
  created_at: string;
  ticker: string;
  id: string;
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ ticker: string; name: string }[]>([]);
  const [ticker, setTicker] = useState("");
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [shouldSearch, setShouldSearch] = useState(true);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [error, setError] = useState("");

  // Fetch ticker suggestions while typing
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (query.length > 1 && shouldSearch) {
        setLoadingSearch(true);
        fetch(`/api/search-ticker?q=${encodeURIComponent(query)}`)
          .then((res) => res.json())
          .then((data) => {
            setResults(data.results || []);
            setLoadingSearch(false);
          })
          .catch(() => setLoadingSearch(false));
      } else {
        setResults([]);
      }
    }, 300);

    return () => clearTimeout(timeout);
  }, [query, shouldSearch]);

  const handleAnalyze = async () => {
    if (!ticker) {
      setError("Please select a ticker.");
      return;
    }
    setError("");
    setLoadingAnalyze(true);
    setAnalysis(null);
    try {
      const res = await fetch(`/api/analyze-and-wait`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company_name: query, ticker }),
      });
      if (!res.ok) throw new Error("Failed to analyze");
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoadingAnalyze(false);
    }
  };

  return (
    <div className="min-h-screen w-full p-6 bg-gray-50 dark:bg-gray-950 flex flex-col">
      <h1 className="text-3xl font-bold mb-6 text-center">Credit Risk Analysis</h1>

      {/* Search Bar */}
      <div className="relative mb-6 w-full max-w-3xl mx-auto">
        <Input
          type="text"
          placeholder="Search for a company or ticker..."
          value={query}
          onChange={(e) => {
            setShouldSearch(true);
            setQuery(e.target.value);
          }}
        />
        {shouldSearch && loadingSearch && (
          <p className="absolute right-3 top-3 text-gray-400 text-sm">Loading...</p>
        )}
        {shouldSearch && results.length > 0 && (
          <ul className="absolute z-10 bg-white border rounded shadow mt-1 w-full max-h-60 overflow-y-auto">
            {results.map((item, index) => (
              <li
                key={`${item.ticker}-${index}`}
                className="p-3 hover:bg-gray-100 cursor-pointer"
                onClick={() => {
                  setQuery(item.name);
                  setTicker(item.ticker);
                  setResults([]);
                  setShouldSearch(false);
                }}
              >
                <span className="font-bold">{item.ticker}</span> - {item.name}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Analyze Button */}
<div className="flex flex-col items-center mb-6">
  <Button onClick={handleAnalyze} disabled={loadingAnalyze}>
    {loadingAnalyze ? (
      <>
        <Loader2 className="animate-spin h-4 w-4 mr-2" /> Analyzing...
      </>
    ) : (
      "Analyze"
    )}
  </Button>

  {/* Info message (always visible) */}
  <p className={`mt-2 text-sm ${loadingAnalyze ? "text-gray-600" : "hidden"}`}>
    This may take around 10 seconds to complete.
  </p>
</div>

      {error && <p className="text-red-500 text-center mb-4">{error}</p>}

      {/* Full-width dashboard */}
      <div className="flex-1 w-full">
        {analysis && <AnalysisDashboard payload={analysis} />}
      </div>
    </div>
  );
}
