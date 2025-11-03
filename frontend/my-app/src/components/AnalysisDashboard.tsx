"use client";
import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar,
} from "recharts";
import { TrendingUp, Sun, Moon, Info, Newspaper, Activity } from "lucide-react";

/**
 * ==========================
 * Types
 * ==========================
 */

export interface FusionWeights {
  structured_expert: number;
  news_sentiment_expert: number;
}

export interface NewsImpactItem {
  headline: string;
  sentiment: "Positive" | "Negative" | "Neutral";
  impact_level: "High" | "Moderate" | "Low";
  sentiment_score: number;
}

export interface TopDriverItem {
  value: string;
  feature: string;
  interpretation: string;
  raw_contribution: number;
  impact_on_final_score: number;
  contribution_percentage: number;
}

export interface ExplainabilityReport {
  company: string;
  final_score: number;
  credit_grade: string;
  data_sources: string[];
  key_insights: {
    concerns: string[];
    strengths: string[];
    market_influence: string;
  };
  risk_summary: {
    risk_level: string;
    description: string;
  };
  assessment_date: string;
  market_conditions: {
    vix: number;
    alpha: number;
    regime: string;
    status: string;
  };
  technical_details: {
    ml_model: string;
    fusion_method: string;
    sentiment_model: string;
    articles_analyzed: number;
    financial_metrics_used: number;
  };
  component_analysis: {
    weights: FusionWeights;
    structured_score: number;
    unstructured_score: number;
  };
  top_financial_drivers: TopDriverItem[];
  news_impact: NewsImpactItem[];
}

export interface CompanyInfo {
  company: string; // "NETFLIX INC (NFLX)"
  credit_grade: string; // "A+"
  analysis_date: string; // "2025-08-22"
  structured_score: string; // "92.5/100"
  final_fused_score: string; // "77.3/100"
  unstructured_score: string; // "51.3/100"
}

export interface HistoricalEntry {
  date: string;
  weights: FusionWeights;
  final_score: number;
  credit_grade: string;
  structured_score: number;
  unstructured_score: number;
}

export interface HistoricalDetailed {
  [key: string]: HistoricalEntry;
}

export interface RootReport {
  company_info: CompanyInfo;
  explainability_report: ExplainabilityReport;
  fusion_explanation: any;
  historical_scores_detailed: HistoricalDetailed;
}



const clamp = (n: number, min = 0, max = 100) => Math.max(min, Math.min(max, n));

const scoreTone = (score: number) => {
  if (score >= 80) return "text-emerald-600";
  if (score >= 60) return "text-amber-600";
  return "text-rose-600";
};

const riskTone = (risk: string) => {
  const text = risk.toUpperCase();
  if (text.includes("LOW")) return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
  if (text.includes("MODERATE")) return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300";
  return "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300";
};

const sentimentTone = (s: string) =>
  s === "Positive"
    ? "text-emerald-600"
    : s === "Negative"
    ? "text-rose-600"
    : "text-slate-600";

/**
 * ==========================
 * Main Component
 * ==========================
 */

interface AnalysisDashboardProps {
  /** The raw server response object you pasted in the prompt. */
  payload: {
    report: string; // JSON string inside the server response
    created_at: string;
    ticker: string;
    id: string;
  };
}

export default function AnalysisDashboard({ payload }: AnalysisDashboardProps) {
  // parse the nested JSON string `payload.report`
  const parsed: RootReport = useMemo(() => {
  if (typeof payload.report === "string") {
    try {
      return JSON.parse(payload.report);
    } catch (e) {
      console.error("Failed to parse report:", e);
      return {} as RootReport;
    }
  }
  return payload.report as RootReport; // If it's already an object
}, [payload.report]);

const { company_info, explainability_report, historical_scores_detailed } = parsed;

  // Dark mode toggle (local)
  const [dark, setDark] = useState<boolean>(false);
  useEffect(() => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [dark]);

  // Historical line data
  const historicalData = useMemo(() => {
    const entries = Object.values(historical_scores_detailed);
    // sort by date ascending for a nice line
    return entries
      .slice()
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((h) => ({
        date: h.date,
        Final: clamp(h.final_score),
        Structured: clamp(h.structured_score),
        Sentiment: clamp(h.unstructured_score),
        W_struct: Math.round(h.weights.structured_expert * 100),
        W_sent: Math.round(h.weights.news_sentiment_expert * 100),
      }));
  }, [historical_scores_detailed]);

  // Weights for pie chart
  const weightsData = useMemo(() => {
    const w = explainability_report.component_analysis.weights;
    return [
      { name: "Structured", value: w.structured_expert },
      { name: "News/Sentiment", value: w.news_sentiment_expert },
    ];
  }, [explainability_report]);

  // Radial for current structured/unstructured scores
  const radialData = useMemo(() => {
    const s = explainability_report.component_analysis.structured_score;
    const u = explainability_report.component_analysis.unstructured_score;
    return [
      { name: "Structured", value: clamp(s) },
      { name: "Sentiment", value: clamp(u) },
    ];
  }, [explainability_report]);
 const radialDataFinal = useMemo(() => {
    const s = explainability_report.final_score;
    return [
      { name: "Final", value: clamp(s) },
    ];
  }, [explainability_report]);

  // Drivers bar data (Top 8)
  const driverBars = useMemo(() => {
    return explainability_report.top_financial_drivers.slice(0, 8).map((d) => ({
      feature: d.feature,
      Impact: d.impact_on_final_score,
    }));
  }, [explainability_report]);

  const finalScore = explainability_report.final_score;

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-50 to-white dark:from-slate-950 dark:to-slate-900 text-slate-900 dark:text-slate-100 p-6">
      <div className="w-full mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight flex items-center gap-2">
              {company_info.company}
              <Badge variant="outline" className="rounded-2xl text-2xl !border-gray-500">
                {company_info.credit_grade}
              </Badge>
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              Assessment time: {new Date(payload.created_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} UTC
            </p>
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle>Final Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-4xl font-bold ${scoreTone(finalScore)}`}>{finalScore.toFixed(2)}</div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Dynamic Weighted Fusion</p>
              <div className="mt-4 h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart innerRadius="40%" outerRadius="100%" data={radialDataFinal}>
                    <RadialBar dataKey="value" cornerRadius={10} startAngle={0} endAngle={270}/>
                    <Tooltip />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle>Component Scores</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  Structured
                  <div className="text-lg font-semibold">
                    {explainability_report.component_analysis.structured_score.toFixed(2)}
                  </div>
                </div>
                <div>
                  Sentiment
                  <div className="text-lg font-semibold">
                    {explainability_report.component_analysis.unstructured_score.toFixed(2)}
                  </div>
                </div>
              </div>
              <div className="mt-4 h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart innerRadius="20%" outerRadius="100%" data={radialData} startAngle={90} endAngle={450}>
                    <RadialBar dataKey="value" cornerRadius={8} />
                    <Tooltip />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle>Model Weights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Structured vs News/Sentiment
              </div>
              <div className="mt-4 h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={weightsData} dataKey="value" nameKey="name" outerRadius={70} label />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid grid-cols-3 w-full mb-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="historical">Historical</TabsTrigger>
          </TabsList>

          {/* Overview */}
          <TabsContent value="overview">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="rounded-2xl shadow-sm">
                <CardHeader>
                  <CardTitle>Risk Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <Badge className={`px-3 py-1 ${riskTone(explainability_report.risk_summary.risk_level)} rounded-2xl`}>
                    {explainability_report.risk_summary.risk_level}
                  </Badge>
                  <p className="mt-2 text-sm">{explainability_report.risk_summary.description}</p>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs uppercase text-slate-500 dark:text-slate-400">Strengths</p>
                      <ul className="mt-1 list-disc pl-5 text-emerald-700 dark:text-emerald-300">
                        {explainability_report.key_insights.strengths.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-xs uppercase text-slate-500 dark:text-slate-400">Concerns</p>
                      <ul className="mt-1 list-disc pl-5 text-rose-700 dark:text-rose-300">
                        {explainability_report.key_insights.concerns.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="rounded-2xl shadow-sm">
                <CardHeader>
                  <CardTitle>Market Conditions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">Status</div>
                      <div className="font-semibold flex items-center gap-2">
                        <Activity size={16} /> {explainability_report.market_conditions.status}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">Regime</div>
                      <div className="font-semibold">{explainability_report.market_conditions.regime}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">VIX</div>
                      <div className="font-semibold">{explainability_report.market_conditions.vix.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 dark:text-slate-400">Alpha</div>
                      <div className="font-semibold">{explainability_report.market_conditions.alpha.toFixed(3)}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 rounded-2xl shadow-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Newspaper size={18} /> News Impact
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="divide-y divide-slate-200 dark:divide-slate-800">
                    {explainability_report.news_impact.length > 0 ? explainability_report.news_impact.map((n, i) => (
                      <div key={i} className="py-3">
                        <div className="font-medium">{n.headline}</div>
                        <div className={`text-sm ${sentimentTone(n.sentiment)}`}>
                          {n.sentiment} • Impact: {n.impact_level} • Score: {n.sentiment_score}
                        </div>
                      </div>
                    )) : <span className="text-gray-500">No latest News found</span>}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Advanced */}
          <TabsContent value="advanced">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <Card className="lg:col-span-2 rounded-2xl shadow-sm">
                <CardHeader>
                  <CardTitle>Top Financial Drivers (Impact on Final Score)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={driverBars} margin={{ left: 12, right: 12 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="feature" hide />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="Impact" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-3 text-xs text-slate-500 dark:text-slate-400">Showing top {driverBars.length} drivers</div>
                </CardContent>
              </Card>

              <Card className="rounded-2xl shadow-sm">
                <CardHeader>
                  <CardTitle>Technical Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="text-sm space-y-1">
                    <li><span className="text-slate-500 dark:text-slate-400">ML Model:</span> {explainability_report.technical_details.ml_model}</li>
                    <li><span className="text-slate-500 dark:text-slate-400">Fusion:</span> {explainability_report.technical_details.fusion_method}</li>
                    <li><span className="text-slate-500 dark:text-slate-400">Sentiment:</span> {explainability_report.technical_details.sentiment_model}</li>
                    <li><span className="text-slate-500 dark:text-slate-400">Articles:</span> {explainability_report.technical_details.articles_analyzed}</li>
                    <li><span className="text-slate-500 dark:text-slate-400">Financial Metrics:</span> {explainability_report.technical_details.financial_metrics_used}</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Historical */}
          <TabsContent value="historical">
            <Card className="rounded-2xl shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><TrendingUp size={18} /> Score Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historicalData} margin={{ left: 12, right: 12 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Final" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="Structured" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="Sentiment" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 flex md:flex-row flex-col gap-3 text-xs text-slate-500 dark:text-slate-400">
                  {historicalData.map((d) => (
                    <div key={d.date} className="flex items-center justify-between border rounded-xl p-2 dark:border-slate-800">
                      <div className="font-medium">{d.date}</div>
                      <div className="flex gap-3">
                        <span>Final: <span className={`font-semibold ${scoreTone(d.Final)}`}>{d.Final.toPrecision(6)}</span></span>
                        <span>W(S): {d.W_struct}%</span>
                        <span>W(N): {d.W_sent}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-center text-xs text-slate-500 dark:text-slate-400"
        >
          Generated at: {new Date(payload.created_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} IST • Ticker: {payload.ticker}
        </motion.div>
      </div>
    </div>
  );
}

