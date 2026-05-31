# AI Investment Analysis Agent - Specification Document

## Project Overview

**Purpose**: Build an interactive AI research assistant for equity valuation using Discounted Cash Flow (DCF) methodology and competitive moat analysis.

**Target User**: Quantitative finance professional seeking to learn AI agent development while building a practical investment research tool.

**Development Approach**: Iterative learning project with cost-effective infrastructure.

---

## 1. Core Capabilities

### 1.1 DCF Valuation Engine
- **Model Structure**: Agent recommends appropriate DCF model (2-stage, 3-stage, or custom) based on:
  - Company maturity (startup, growth, mature)
  - Industry characteristics
  - Historical growth patterns
  - Analyst consensus forecasts

- **Key Assumptions (Hybrid Human-AI Approach)**:
  - Agent calculates and suggests:
    - WACC (using CAPM, cost of debt from financials)
    - Terminal growth rate (based on GDP, industry growth)
    - Revenue growth rates by stage
    - Margin assumptions
  - User reviews and can override any assumption
  - Agent explains rationale for each suggested parameter

- **Sensitivity Analysis**:
  - Multi-dimensional sensitivity tables
  - Key variables: WACC (±2%), terminal growth (±1%), revenue growth (±5%)
  - Monte Carlo simulation option for probabilistic valuation range
  - Visualization of value ranges under different scenarios

### 1.2 Competitive Moat Analysis

**Framework**: Morningstar/Buffett 5 Sources of Economic Moat

1. **Intangible Assets**
   - Brand strength (pricing power indicators)
   - Patents and intellectual property
   - Regulatory licenses/approvals

2. **Switching Costs**
   - Customer retention metrics
   - Integration complexity
   - Data lock-in effects

3. **Network Effects**
   - User growth vs. value creation
   - Platform dynamics
   - Multi-sided market analysis

4. **Cost Advantages**
   - Scale economies (margin trends)
   - Unique assets or resources
   - Process efficiencies

5. **Efficient Scale**
   - Market structure analysis
   - Barriers to entry assessment
   - Competitive response likelihood

**Quantitative Indicators**:
- ROIC trends (5-10 year history)
- ROE vs. industry average
- Gross and operating margin stability
- Pricing power proxy (price increases vs. volume impact)
- Customer acquisition cost trends
- Market share evolution

**Qualitative Analysis**:
- Industry structure assessment (Porter's 5 Forces)
- Regulatory barriers
- Technology disruption risk
- Management quality indicators

---

## 2. Technical Architecture

### 2.1 LangGraph Agent Design

**State Schema**:
```python
class InvestmentAnalysisState(TypedDict):
    # Input
    ticker_symbols: List[str]
    user_overrides: Dict[str, Any]
    
    # Data Collection
    financial_data: Dict[str, pd.DataFrame]
    market_data: Dict[str, Any]
    
    # Analysis Components
    dcf_assumptions: Dict[str, Any]
    dcf_results: Dict[str, Any]
    moat_analysis: Dict[str, Any]
    
    # User Interaction
    conversation_history: List[Dict]
    pending_approvals: List[str]
    
    # Output
    final_report: str
    sensitivity_tables: Dict
    visualizations: List
```

**Agent Graph Nodes**:

1. **Data Acquisition Node**
   - Free API integration (Alpha Vantage, Yahoo Finance, FRED)
   - Manual data input handler
   - Data validation and cleaning
   - Caching layer (SQLite or pickle files)

2. **Company Analysis Node**
   - Determine company lifecycle stage
   - Industry classification
   - Historical performance trends
   - Recommend DCF model structure

3. **DCF Calculator Node**
   - WACC calculation (beta from regression, debt costs)
   - Free cash flow projection
   - Terminal value calculation
   - Present value summation
   - Per-share intrinsic value

4. **Assumption Review Node**
   - Present assumptions to user
   - Wait for approval/modifications
   - Update calculations with user inputs
   - Document rationale for final assumptions

5. **Sensitivity Analysis Node**
   - Generate sensitivity tables
   - Create tornado diagrams
   - Calculate probability-weighted scenarios
   - Identify key value drivers

6. **Moat Analyzer Node**
   - Calculate quantitative moat metrics
   - Fetch industry data for benchmarking
   - Generate qualitative assessment prompts
   - Score moat strength (Wide/Narrow/None)

7. **Report Generator Node**
   - Detailed written analysis
   - Structured JSON/CSV output
   - Interactive visualizations
   - Investment thesis summary

8. **Conversation Manager Node**
   - Handle user questions
   - Allow drill-down into specific sections
   - Support "what-if" scenario testing
   - Maintain context across interactions

**Conditional Edges**:
- User approval gates before proceeding
- Data availability checks (API vs. manual input)
- Multi-stock iteration logic
- Error handling and retry mechanisms

### 2.2 LLM Integration

**Primary Model**: GPT-4o mini
- **Cost**: ~$0.15/$0.60 per million tokens (input/output)
- **Use Cases**: 
  - Natural language understanding of user requests
  - Assumption reasoning and explanation
  - Qualitative moat analysis
  - Report generation

**Prompt Engineering Strategy**:
- System prompts with finance domain knowledge
- Few-shot examples for DCF calculation explanations
- Chain-of-thought prompting for moat assessment
- Structured output schemas (JSON mode)

**Token Optimization**:
- Cache financial data locally, only summarize for LLM
- Use function calling for structured outputs
- Batch similar analyses in single prompt
- Stream responses for better UX

### 2.3 Data Sources

**Free APIs** (with fallback hierarchy):
1. **yfinance** (Yahoo Finance): Historical prices, basic fundamentals
2. **Alpha Vantage**: Income statement, balance sheet, cash flow (500 calls/day free)
3. **FRED API**: Risk-free rate, GDP growth, economic indicators
4. **SEC EDGAR**: 10-K/10-Q filings (direct HTML parsing)

**Manual Input Interface**:
- CSV upload for custom financial projections
- Form-based input for key assumptions
- Excel template integration

**Caching Strategy**:
- SQLite database for historical financials (update quarterly)
- 24-hour cache for market data (prices, rates)
- Permanent storage for user-provided data
- Cache invalidation rules

---

## 3. User Interface - Jupyter Notebook

### 3.1 Notebook Structure

**Cell 1: Setup & Configuration**
```python
# API keys, model selection, imports
# One-time configuration
```

**Cell 2: Agent Initialization**
```python
# Initialize LangGraph agent
# Load cached data
# Display available tickers
```

**Cell 3: Analysis Request**
```python
# User specifies: ticker(s), analysis type
# Initiate agent workflow
```

**Cell 4: Interactive Review**
```python
# Display assumptions
# Accept user modifications
# Re-run calculations
```

**Cell 5: Results Display**
```python
# Tabular results
# Visualizations (matplotlib/plotly)
# Sensitivity charts
```

**Cell 6: Conversational Q&A**
```python
# Ask follow-up questions
# Request specific deep-dives
# Compare stocks
```

**Cell 7: Export**
```python
# Save reports (PDF, HTML, JSON)
# Export data (CSV, Excel)
```

### 3.2 Visualization Components

- **DCF Waterfall Chart**: Show value buildup
- **Sensitivity Tornado Diagram**: Key driver impacts
- **Moat Scorecard**: Radar chart of 5 moat dimensions
- **Time Series**: ROIC, margins, growth rates
- **Comparable Multiples**: Industry positioning

---

## 4. Detailed Functional Requirements

### 4.1 DCF Calculation Workflow

**Step 1: Data Gathering**
- Fetch 5 years historical financials
- Retrieve current stock price and shares outstanding
- Get risk-free rate (10Y Treasury) and market risk premium

**Step 2: WACC Calculation**
- **Cost of Equity (CAPM)**:
  - Beta: 5-year monthly regression vs. S&P 500
  - Risk-free rate: Current 10Y Treasury yield
  - Market risk premium: 7% (configurable)
- **Cost of Debt**:
  - Interest expense / Total debt
  - Tax shield: Effective tax rate from financials
- **Capital Structure**: Market value weights

**Step 3: Free Cash Flow Projection**
- **Historical Analysis** (5 years):
  - Revenue CAGR
  - Operating margin trends
  - CapEx as % of revenue
  - NWC changes
- **Forecast Assumptions** (agent suggests, user approves):
  - Revenue growth by stage
  - Terminal growth rate (≤ GDP + inflation)
  - Margin improvement/degradation path
  - Reinvestment requirements

**Step 4: Terminal Value**
- Gordon Growth Model: FCF_terminal / (WACC - g)
- Exit multiple method (as sanity check)

**Step 5: Intrinsic Value**
- Discount all cash flows to present
- Add cash, subtract debt
- Divide by shares outstanding
- Compare to current price (margin of safety)

### 4.2 Moat Analysis Workflow

**Phase 1: Quantitative Scoring** (0-10 scale per dimension)

1. **Intangible Assets** (0-10):
   - Brand: Gross margin vs. competitors (+2 if >10% higher)
   - Patents: R&D/Revenue ratio (+2 if >5%)
   - Pricing power: 3-year price CAGR vs. volume CAGR (+3 if price > volume)

2. **Switching Costs** (0-10):
   - Customer retention: Annual churn rate (+4 if <5%)
   - Contract length: Revenue from multi-year contracts (+3 if >50%)
   - Net revenue retention (+3 if >110%)

3. **Network Effects** (0-10):
   - User growth acceleration (+3 if accelerating)
   - Marketplace balance (+2 if supply ≈ demand growth)
   - Platform revenue per user (+5 if increasing)

4. **Cost Advantages** (0-10):
   - Operating leverage: Margin expansion with revenue growth (+4)
   - Unit economics: Customer LTV/CAC ratio (+3 if >3x)
   - Asset turns: Revenue/Assets vs. industry (+3 if higher)

5. **Efficient Scale** (0-10):
   - Market concentration: HHI index (+5 if >2500)
   - Regulatory barriers (+3 if high)
   - Capacity utilization (+2 if >85%)

**Phase 2: Qualitative Assessment** (LLM-generated)
- Industry structure narrative
- Competitive threats identification
- Moat sustainability over 10 years
- Key risks to moat erosion

**Phase 3: Overall Moat Rating**
- **Wide Moat**: Total score >35, sustained ROIC >15%
- **Narrow Moat**: Total score 20-35, ROIC >12%
- **No Moat**: Score <20 or declining ROIC

### 4.3 Sensitivity Analysis

**Approach 1: Data Tables**
- 2D tables: WACC (rows) × Terminal Growth (columns)
- 3D scenario: Add revenue growth as third dimension
- Output: Intrinsic value matrix

**Approach 2: Monte Carlo Simulation**
- Define probability distributions:
  - WACC: Normal (mean=calculated, σ=1%)
  - Terminal growth: Triangular (min=0%, mode=2%, max=4%)
  - Revenue growth: Lognormal (fit to historical)
- Run 10,000 simulations
- Output: Probability distribution of intrinsic value
- Calculate Value at Risk (VaR) metrics

**Visualization**:
- Tornado chart: Absolute value change per variable
- Probability histogram: Distribution of outcomes
- Percentile table: 10th, 25th, 50th, 75th, 90th percentiles

---

## 5. Implementation Roadmap

### Phase 1: MVP (Weeks 1-2)
**Goal**: Single stock DCF with manual inputs

- [ ] Set up development environment (Python, LangGraph, Jupyter)
- [ ] Create basic state schema
- [ ] Build simple linear graph: Input → Calculate → Display
- [ ] Implement WACC calculation
- [ ] Build 2-stage DCF model
- [ ] Create basic notebook interface
- [ ] Test with 2-3 known stocks (manual data entry)

**Deliverable**: Working DCF calculator with GPT-4o mini explanations

### Phase 2: Data Integration (Weeks 3-4)
**Goal**: Automated data fetching and caching

- [ ] Integrate yfinance for price/basic data
- [ ] Add Alpha Vantage for detailed financials
- [ ] Implement FRED API for macro data
- [ ] Build SQLite caching layer
- [ ] Add data validation and error handling
- [ ] Create fallback logic (API → manual)

**Deliverable**: Fully automated data pipeline

### Phase 3: Moat Analysis (Weeks 5-6)
**Goal**: Comprehensive competitive advantage assessment

- [ ] Implement quantitative moat metrics
- [ ] Build industry benchmarking
- [ ] Create LLM prompts for qualitative analysis
- [ ] Design moat scoring algorithm
- [ ] Add visualization (radar chart)
- [ ] Test against Morningstar moat ratings

**Deliverable**: Integrated DCF + Moat analysis

### Phase 4: Interactivity (Weeks 7-8)
**Goal**: Human-in-the-loop assumption review

- [ ] Add assumption presentation node
- [ ] Implement user override mechanism
- [ ] Build conversational Q&A handler
- [ ] Create what-if scenario testing
- [ ] Add memory across sessions
- [ ] Implement multi-stock comparison

**Deliverable**: Interactive research assistant

### Phase 5: Sensitivity & Reporting (Weeks 9-10)
**Goal**: Complete analysis with sensitivity and polished outputs

- [ ] Build sensitivity analysis engine
- [ ] Add Monte Carlo simulation
- [ ] Create visualization suite
- [ ] Implement report generation (markdown, HTML, PDF)
- [ ] Add structured data export (JSON, CSV)
- [ ] Polish notebook UX

**Deliverable**: Production-ready investment analysis tool

### Phase 6: Optimization (Week 11+)
**Goal**: Refine, optimize, and extend

- [ ] Optimize token usage (reduce costs)
- [ ] Add unit tests and validation
- [ ] Benchmark against professional tools
- [ ] Document code and create tutorials
- [ ] Consider web deployment (Streamlit)
- [ ] Explore Claude 3.5 Haiku comparison

---

## 6. Cost Projections

### Development Phase Costs (10 weeks)

**Assumptions**:
- 50 test analyses per week
- Average 10K input tokens + 5K output tokens per analysis
- GPT-4o mini pricing: $0.15 input / $0.60 output per 1M tokens

**Weekly Cost**:
- Input: 50 × 10K × $0.15/1M = $0.075
- Output: 50 × 5K × $0.60/1M = $0.15
- **Total: ~$0.23/week**

**10-Week Development**: ~$2.30

**API Call Costs** (Alpha Vantage premium if needed):
- Free tier: 500 calls/day (sufficient for learning)
- Premium: $50/month (only if scaling up)

**Total Learning Project Cost**: <$5 for 10 weeks

### Production Use Costs (post-development)

**Light Use** (10 analyses/month):
- ~$0.05/month

**Moderate Use** (100 analyses/month):
- ~$0.45/month

**Heavy Use** (500 analyses/month):
- ~$2.25/month

**Conclusion**: Extremely cost-effective for learning and personal use.

---

## 7. Success Metrics

### Learning Objectives
- [ ] Understand LangGraph state management
- [ ] Master function calling and tool use
- [ ] Build end-to-end agentic application
- [ ] Optimize LLM costs and performance

### Functional Validation
- [ ] DCF values within ±10% of Bloomberg/professional tools
- [ ] Moat ratings align with Morningstar >70% of the time
- [ ] Sensitivity ranges cover realistic scenarios
- [ ] User can complete analysis in <5 minutes
- [ ] Agent explanations are finance-accurate

### Technical Performance
- [ ] API call caching reduces redundant requests by >80%
- [ ] Token usage <15K per analysis
- [ ] Notebook executes without errors
- [ ] Response time <30 seconds per node

---

## 8. Extension Ideas (Future Versions)

### Version 2.0 Enhancements
- **Relative Valuation**: Add P/E, EV/EBITDA multiples analysis
- **Dividend Discount Model**: Alternative valuation for dividend stocks
- **Sum-of-the-Parts**: Multi-segment company valuation
- **Credit Analysis**: Bond investor perspective

### Version 3.0 Advanced Features
- **Automated Financial Statement Parsing**: Direct 10-K analysis
- **Industry Peer Auto-Discovery**: Automatic comp selection
- **News Sentiment Integration**: Impact on moat/assumptions
- **Portfolio Optimization**: Multi-stock allocation

### Model Upgrades
- **A/B Testing**: Compare GPT-4o mini vs. Claude 3.5 Haiku
- **Specialized Models**: Use Claude Sonnet for complex moat reasoning
- **Ensemble Approach**: Combine multiple model outputs

---

## 9. Risk Mitigation

### Technical Risks
- **API Rate Limits**: Use caching, implement exponential backoff
- **Data Quality**: Validate against multiple sources, flag anomalies
- **LLM Hallucinations**: Ground in retrieved data, use structured outputs
- **Cost Overruns**: Set usage quotas, implement token budgets

### Financial Analysis Risks
- **Model Risk**: Document assumptions, show sensitivity
- **Data Lag**: Clearly timestamp all data sources
- **Overfitting**: Validate against out-of-sample companies
- **Bias**: Disclose agent limitations in report

### Compliance Considerations
- **Not Financial Advice**: Clear disclaimers in outputs
- **Data Licensing**: Respect API terms of service
- **Privacy**: Don't expose proprietary research to LLM providers

---

## 10. Resources & References

### LangGraph Learning
- Official LangGraph documentation
- LangChain Academy tutorials
- GitHub examples: financial agents

### Finance Frameworks
- **DCF**: Damodaran (NYU Stern) valuation materials
- **Moat**: Morningstar Economic Moat methodology
- **Competitive Strategy**: Porter's Five Forces

### Python Libraries
- **LangGraph**: Agent orchestration
- **OpenAI**: LLM API
- **yfinance**: Market data
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **matplotlib/plotly**: Visualization
- **requests**: API calls
- **sqlite3**: Caching

### APIs
- Alpha Vantage: https://www.alphavantage.co/
- FRED: https://fred.stlouisfed.org/docs/api/
- SEC EDGAR: https://www.sec.gov/edgar/sec-api-documentation

---

## 11. Getting Started Checklist

### Pre-Development
- [ ] Install Python 3.10+
- [ ] Set up virtual environment
- [ ] Obtain OpenAI API key
- [ ] Register for Alpha Vantage API key (free)
- [ ] Clone/create project repository
- [ ] Install Jupyter Lab

### Week 1 Tasks
- [ ] Review LangGraph quickstart tutorial
- [ ] Build "Hello World" agent
- [ ] Manually calculate DCF for one stock (e.g., AAPL)
- [ ] Define state schema in code
- [ ] Create initial notebook structure

### First Analysis Goal
**Target**: Analyze Apple (AAPL) with manual data input, producing:
1. Intrinsic value estimate
2. Simple 2-stage DCF
3. WACC calculation with explanation
4. Text output of key assumptions

**Success**: Match your manual calculation, with agent explaining each step.

---

## Appendix A: Sample State Schema (Python)

```python
from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

class FinancialData(TypedDict):
    """Financial data for a single company"""
    ticker: str
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    historical_prices: pd.DataFrame
    last_updated: datetime

class DCFAssumptions(TypedDict):
    """User-approved DCF assumptions"""
    wacc: float
    terminal_growth_rate: float
    revenue_growth_rates: List[float]  # By year/stage
    operating_margin_target: float
    capex_percent_revenue: float
    tax_rate: float
    model_type: str  # "2-stage", "3-stage", etc.

class MoatMetrics(TypedDict):
    """Quantitative moat indicators"""
    roic_5yr_avg: float
    roe_vs_industry: float
    gross_margin_vs_peers: float
    customer_retention_rate: Optional[float]
    pricing_power_score: float

class InvestmentAnalysisState(TypedDict):
    """Complete agent state"""
    # Inputs
    ticker_symbols: List[str]
    analysis_date: datetime
    
    # Retrieved data
    financial_data: Dict[str, FinancialData]
    market_data: Dict[str, Any]  # Risk-free rate, market premium, etc.
    
    # Analysis outputs
    dcf_assumptions: Dict[str, DCFAssumptions]
    dcf_results: Dict[str, Dict[str, float]]  # ticker -> {intrinsic_value, current_price, upside}
    moat_analysis: Dict[str, Dict[str, Any]]
    sensitivity_results: Dict[str, pd.DataFrame]
    
    # Interaction
    conversation_history: List[Dict[str, str]]
    pending_user_approval: Optional[str]  # Node waiting for input
    user_overrides: Dict[str, Any]
    
    # Outputs
    final_reports: Dict[str, str]  # ticker -> markdown report
    visualizations: Dict[str, Any]
```

---

## Appendix B: Example Moat Analysis Output

```markdown
### Economic Moat Analysis: Apple Inc. (AAPL)

**Overall Moat Rating**: WIDE MOAT ⭐⭐⭐⭐⭐

**Quantitative Score**: 42/50

#### 1. Intangible Assets (9/10) ⭐⭐⭐⭐⭐
- **Brand Strength**: Premium pricing sustained (iPhone ASP +15% vs. Android avg)
- **Ecosystem Lock-in**: 2B+ active devices, 935M paying subscribers
- **Intellectual Property**: 50K+ patents, design differentiation

#### 2. Switching Costs (8/10) ⭐⭐⭐⭐
- **Customer Retention**: 92% iPhone loyalty (vs. 74% Android)
- **Data/Service Lock-in**: iCloud, Apple Music, iMessage network effects
- **Hardware Integration**: Seamless device ecosystem (Mac, iPad, Watch, AirPods)

#### 3. Network Effects (7/10) ⭐⭐⭐⭐
- **App Store**: 1.8M apps, developer platform moat
- **Services Ecosystem**: Bundled offerings (Apple One) increase stickiness
- **Two-sided marketplace**: Developers + users create reinforcing loop

#### 4. Cost Advantages (9/10) ⭐⭐⭐⭐⭐
- **Scale Economics**: Supply chain dominance, component exclusivity
- **Vertical Integration**: Custom silicon (A-series, M-series chips)
- **Operating Leverage**: GM expanding despite flat units (44% → 46%)

#### 5. Efficient Scale (9/10) ⭐⭐⭐⭐⭐
- **Market Position**: 50%+ premium smartphone profit share
- **Barriers to Entry**: $100B+ R&D required to replicate ecosystem
- **Regulatory Moat**: App Store policies defend margin (under pressure)

**Sustainability Assessment** (10-year horizon):
- **Strengths**: Ecosystem lock-in intensifying, services revenue growing faster than hardware
- **Risks**: Regulatory scrutiny (EU DMA, US antitrust), China geopolitical exposure
- **Verdict**: Moat remains wide but faces headwinds; pricing power intact

**ROIC Validation**: 
- 5-year average ROIC: 44% (vs. Tech sector avg: 18%)
- Trend: Stable, indicating durable competitive advantages

**Investment Implication**: 
Premium valuation justified by moat width. DCF should use lower discount rate (moat risk premium).
```

---

**End of Specification Document**

*This living document should be updated as you learn and iterate. Good luck with your first AI agent!*

##### To skip permission use the following command:
claude --dangerously-skip-permissions