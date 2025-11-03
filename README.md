# BlackBox Cred - Explainable Credit Scorecard System

BlackBox Cred is an advanced credit risk assessment platform that combines structured financial data, unstructured news data, and macroeconomic indicators to provide comprehensive and explainable credit scores for companies.

To use the app press: https://black-box-cred.vercel.app/ 

Here’s a visual overview of of our Pileline:

![Project Pipeline](ModelArchitecture_BlackBox.svg)


##  Core Features 

### 1. Multi-Source Data Integration
- **Financial Data**: Balance sheets, income statements, cash flows from Yahoo Finance
- **Market Data**: Stock prices, volatility, trading volumes
- **Macro Indicators**: Interest rates, GDP growth, unemployment from FRED
- **News & Sentiment**: Real-time news analysis and sentiment scoring

### 2. Custom Feature Engineering
- **Enhanced Z-Score Model**
  - Modified Altman Z-score with dynamic weights
  - Incorporates market-based indicators
  - Sector-specific adjustments


    $$
\text{Enhanced Z} = 1.5X_1 + 1.6X_2 + 3.5X_3 + 0.8X_4 + 1.2X_5
$$

**Where:**  
- **X₁** = Dynamic Liquidity Stress Index  
- **X₂** = Multi-Period Earnings Quality Score  
- **X₃** = Risk-Adjusted Operational Performance  
- **X₄** = Multi-Dimensional Solvency Score  
- **X₅** = Asset Efficiency Index  

**Scaled to 0-10.**

  
- **KMV Distance-to-Default**
  - Option-theoretic approach to default prediction
  - Market-based asset volatility estimation
  - Dynamic default barrier calculation

$$
D = \frac{\ln\left(\frac{A}{DP}\right) + \left(\mu - \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}
$$

**Where:**  
- **A** = Equity Value + Debt Value  
- **DP** = Short-Term Debt + 0.5 × Long-Term Debt  
- **σ** = Volatility · leverage-adjusted  
- **T** = 1 yr  

**Output clamped to:**[-5, 10]
  

### 3. Explainable AI Components
- Feature importance visualization
- Risk factor contribution analysis
- Interactive scenario testing
- Confidence interval estimations

### 4. Caching
- Query results are stored in Neon Postgres for fast access
- Automatic cache check on "analyze-and-wait" endpoint of fastAPI
- Instant retrieval from cache for entries fresher than six hours
- Run full pipeline and refresh cache when entry is expired


### 5. Scoring Engines

- **Structured Data**
  
  - ### **EBM – Explainable Boosting Machine**
    - Uses metrics including KMV Distance-to-Default, Enhanced Z-Score and additional proprietary measures.
    - Interpretable additive model  
    - Glass-box additive learner: keeps full interpretability while still fitting non-linear, high-order patterns  
    - **Feature Set:** 34 numeric inputs  
    - **Probability of Investment Grade** scaled to get a score between 0 and 100

- **Unstructured Data**
  - ### **FinBERT Sentiment Pass**
    - Infers positive, negative, and neutral scores per article  

  - ### **Risk-Keyword Booster**
    - Regex lists for liquidity crisis, debt distress, operational and market risks  
    - Each hit adds **3 pts**; capped at **+25**  
    - **Final score:**
      
$$
100 - \left[50\left(1 - \texttt{avgNetSentiment}\right) + \min\left(\texttt{riskKeywords}\times 3\, 25\right)\right]
$$


### 6. Fusion System
- ### **Two-Expert Weighted Combination**
  - Structured Credit Score + Unstructured Credit Score → single fused score.



- ### **Market Regime Classification**
  - VIX > 30 → **VOLATILE**  
  - VIX > 25 + negative trend → **BEAR**  
  - VIX < 15 + positive trend → **BULL**  
  - Everything else → **NORMAL**



- ### **Dynamic Weight Adjustments**
  - **VOLATILE:** Structured weight ×1.2, News weight ×0.9  
  - **BEAR:** News weight ×1.3 (sentiment matters more)



- ### **Expert Agreement Calculation**
  - Takes both expert scores (structured and unstructured), calculates coefficient of variation.  
  - If agreement < 30%, system favors structured expert by +10%.



- ### **Regime-Based Risk Adjustment**
  - Base adjustment based on Market Context:  
    - **BEAR:** -10  
    - **VOLATILE:** -7  
    - **BULL:** +5  
    - **NORMAL:** 0  
  - Points added based on current market regime.  
  - **Disagreement Amplifier:**

$$
\text{Base adjustment} \times \big[ 1 + (\text{disagreement} \times 0.5) \big]
$$

### API Layer
```
FastAPI Service
     ↓
Request Validation → Processing
     ↑
Response Caching
```

##  Project Structure <to be changed>

```
BlackBox_Cred/
├── fastAPI/                       # API service
│   ├── Dockerfile                 # Container configuration
│   ├── requirements.txt           # Dependencies
│   └── app/
│       ├── alembic.ini            # Database migrations config
│       ├── crud.py                # Database operations
│       ├── db.py                  # Database setup
│       ├── dependencies.py        # API dependencies
│       ├── main.py                # API initialization
│       ├── models.py              # Data models
│       ├── scoring.py             # Scoring logic
│       ├── migrations/            # Database migrations
│       ├── pipeline/              # Core processing pipeline
│       │   ├── data_collection.py # Data collection module
│       │   ├── data_processing.py # Data processing logic
│       │   ├── explainability.py  # Explainability utilities
│       │   ├── fusion_engine.py   # Fusion engine
│       │   ├── main_pipeline.py   # Orchestrating pipeline
│       │   ├── structured_analysis.py   # Structured data analysis
│       │   ├── unstructured_analysis.py # Unstructured data analysis
│       │   └── model/             # Trained models
│       │       └── ebm_model_struct_score.pkl # Pre-trained scoring model
│       └── routes/                # API endpoints
│           └── scores.py          # Scoring endpoint
│
└── frontend/                      # Web interface
    └── my-app/
        ├── components.json        # Component configuration
        ├── next.config.ts         # Next.js configuration
        ├── package.json           # Project dependencies
        ├── postcss.config.mjs     # PostCSS configuration
        ├── tsconfig.json          # TypeScript configuration
        ├── public/                # Static assets
        │   ├── file.svg           # SVG asset
        │   ├── globe.svg          # SVG asset
        │   ├── mock-report.json   # Mock report data
        │   ├── next.svg           # SVG asset
        │   ├── vercel.svg         # SVG asset
        │   └── window.svg         # SVG asset
        └── src/
            ├── app/               # Next.js pages
            │   ├── globals.css    # Global styles
            │   ├── layout.tsx     # Layout configuration
            │   ├── page.tsx       # Main landing page
            │   └── search/
            │       └── page.tsx   # Search page
            ├── components/        # UI components
            │   ├── AnalysisDashboard.tsx # Dashboard view
            │   ├── app-sidebar.tsx       # Sidebar component
            │   ├── TickerSearch.tsx      # Search bar
            │   └── ui/                   # Reusable UI components
            ├── hooks/             # Custom React hooks
            │   └── use-mobile.ts  # Mobile hook
            └── lib/               # Utility functions
                └── utils.ts       # Helper utilities
```

##  Local Development Setup

### Backend Setup

1. **Python Environment Setup**
```bash
# Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Database Setup**
```bash
# Option 1: Local PostgreSQL
createdb blackbox_cred
psql -d blackbox_cred -f schema.sql

# Option 2: NeonDB (Recommended)
# Use the provided NEON_DSN in .env
```

3. **API Keys Configuration**
```bash
# Create .env file

# Add your API keys
FRED_API_KEY=your_fred_key
FINNHUB_API_KEY=your_finnhub_key
NEWS_API_KEY=news_api_key
DATABASE_URL=your_database_url
```

### API Layer Setup

```bash
cd fastAPI
pip install -r requirements.txt

# Development server
uvicorn app.main:app --reload --port 8000

# Production server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Setup

```bash
cd frontend/my-app

# Install dependencies
npm install

# Development server
npm run dev

# .env file

NEXT_PUBLIC_BACKEND_URL=backend_url || http://localhost:8000

# Production build
npm run build
npm start
```
