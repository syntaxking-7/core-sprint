"use client";

import { useEffect, useState } from "react";

export default function TickerSearch({ onSelect }: { onSelect: (ticker: string) => void }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ ticker: string; name: string }[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (query.length > 1) {
        setLoading(true);
        fetch(`/api/search-ticker?q=${encodeURIComponent(query)}`)
          .then((res) => res.json())
          .then((data) => {
            // Your FastAPI response: { results: [{ ticker, name }, ...] }
            setResults(data.results || []);
            setLoading(false);
          })
          .catch(() => setLoading(false));
      } else {
        setResults([]);
      }
    }, 300); // debounce 300ms

    return () => clearTimeout(timeout);
  }, [query]);

  return (
    <div className="relative w-full max-w-md mx-auto">
      <input
        type="text"
        placeholder="Search for a company or ticker..."
        className="w-full p-3 border rounded"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {loading && <p className="absolute right-3 top-3 text-gray-400 text-sm">Loading...</p>}
      {results.length > 0 && (
        <ul className="absolute z-10 bg-white border rounded shadow mt-1 w-full max-h-60 overflow-y-auto">
          {results.map((item) => (
            <li
              key={item.ticker}
              className="p-3 hover:bg-gray-100 cursor-pointer"
              onClick={() => {
                setQuery(item.name);
                setResults([]);
                onSelect(item.ticker);
              }}
            >
              <span className="font-bold">{item.ticker}</span> - {item.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
