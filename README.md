# Post-Disaster CSR Fund Allocation Agent for Louisiana Oil & Gas Companies

**CSC 7644: Applied LLM Development — Final Project**  
**Author:** Tahmida Sarker Muna | Louisiana State University  


---

## Project Overview

This is the final project for **CSC 7644: Applied LLM Development** at Louisiana State University.

Louisiana oil and gas companies regularly commit CSR (Corporate Social Responsibility) budgets to community recovery after major disasters such as hurricanes, floods, and winter storms. Today, allocation decisions rely on proximity to company facilities, media coverage, or informal networks, none of which systematically identifies the communities with the greatest objective need.

This project builds an **agentic LLM application** that turns that subjective process into a transparent, data-driven one. Given a FEMA disaster declaration number as input, the agent automatically identifies all affected Louisiana parishes, retrieves hazard severity and social vulnerability data from federal public APIs, ranks parishes by a weighted composite need score, and generates a plain-language CSR allocation report with every figure cited back to its source.

---

## Key Features / Capabilities

- **Agentic reasoning loop** — GPT-4o with function calling autonomously decides which tools to invoke, in what order, and how to synthesize the results into a final report — no manual intervention required. Falls back to GPT-4o-mini under rate limits.
- **Four live data tools** — Real-time queries to OpenFEMA (disaster declarations), FEMA NRI (hazard risk scores), CDC SVI (social vulnerability), and U.S. Census ACS (poverty and housing data). No paid data sources required.
- **RAG-grounded interpretation** — A FAISS vector store built over FEMA NRI, CDC SVI, and CEJST methodology PDFs ensures every score interpretation is cited from authoritative documentation rather than hallucinated.
- **Weighted need scoring** — A deterministic composite parish ranking: NRI risk (30%), SVI vulnerability (40%), poverty rate (20%), housing cost burden (10%). Normalized to 0–1 before weighting.
- **Streamlit web UI** — Browser-based interface with preset disaster selectors, real-time agent progress, Markdown report rendering, and one-click `.txt` download.
- **Evaluation framework** — Automated Precision@5 scoring against ground-truth parish damage rankings for Hurricane Ida (DR-4611, 2021) and the August 2016 Louisiana floods (DR-4277).

---

## Tech Stack and Architecture

### Components

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o with function calling (GPT-4o-mini fallback) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (`faiss-cpu`) — flat L2 index |
| Back-end Agent | Python (`agent/llm_agent.py`) — custom agentic loop |
| Data APIs | OpenFEMA REST, FEMA NRI ArcGIS REST, CDC SVI local CSV, U.S. Census ACS |
| Front End | Streamlit (`app.py`) |
| PDF Extraction | PyMuPDF (`fitz`) |
| Language | Python 3.11+ |

### High-Level Architecture

```
User Query (FEMA disaster number)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                    CSRAllocationAgent                    │
│  (agent/llm_agent.py)                                    │
│                                                          │
│  System Prompt + Few-shot Example  (agent/prompts.py)    │
│       +                                                  │
│  RAG Context  (rag/ → FAISS → methodology PDFs)          │
│       │                                                  │
│       ▼                                                  │
│  GPT-4o  ── function calling loop ──────────────────┐    │
│                                                     │    │
│  ┌──────────────────────────────────────────────┐   │    │
│  │  Tools  (agent/tools.py)                     │◄──┘    │
│  │  get_affected_parishes  → OpenFEMA API        │        │
│  │  nri_lookup             → FEMA NRI ArcGIS     │        │
│  │  svi_lookup             → CDC SVI local CSV   │        │
│  │  census_lookup          → U.S. Census ACS     │        │
│  └──────────────────────────────────────────────┘        │
│       │                                                   │
│       ▼  tool results injected back into context          │
│  GPT-4o reasons → weighted ranking → allocation report    │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  Streamlit UI  (app.py)  or  Terminal / Script
```

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher (tested on 3.11 and 3.12)
- `pip` or `conda` for package installation
- Windows, macOS, or Linux
- An OpenAI API key — [get one here](https://platform.openai.com/api-keys)

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/csc7644-final-project-sarker.git
cd csc7644-final-project-sarker
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Or using conda
conda create -n csr-agent python=3.11
conda activate csr-agent
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

```bash
cp .env.example .env   # macOS / Linux
copy .env.example .env  # Windows
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-your-key-here
CENSUS_API_KEY=your-census-key-here   # Optional — unauthenticated access works but is rate-limited
```

> The `OPENAI_API_KEY` is the only required key. Without it the agent cannot call the LLM or build embeddings.

---

## Running the Application

### Web application (recommended)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**How to use the web UI:**

1. The OpenAI API key is read automatically from `.env` — no manual entry needed.
2. Use the dropdown to select a preset disaster event (e.g., Hurricane Ida 2021, DR-4611), or type any valid FEMA disaster number.
3. Review or edit the pre-filled query in the text area.
4. Click **Run Analysis**.
5. Wait 30–60 seconds while the agent retrieves live data from four federal APIs and generates the report.
6. Read the ranked parish allocation report rendered in the page.
7. Click **Download Report (.txt)** to save a copy.

### Command-line / Python script

```python
from agent import CSRAllocationAgent

agent = CSRAllocationAgent(verbose=True)

report = agent.run(
    "Our company wants to direct CSR funds to communities most affected "
    "by disaster number 4611 in Louisiana. Please generate a full allocation report."
)
print(report)
```

### Evaluation script

```bash
# Evaluate against Hurricane Ida (2021)
python -m evaluation.eval_runner --disaster 4611

# Evaluate against both configured test events
python -m evaluation.eval_runner --all
```

Output includes Precision@5 score, detected tool calls, latency, and estimated API cost. Full reports are saved to `evaluation/reports/`.

---

## Repository Organization

```
csc7644-final-project-sarker/
│
├── app.py                    # Streamlit front-end — UI entry point
├── requirements.txt          # All Python dependencies with minimum versions
├── .env.example              # Template for environment variables (safe to commit)
├── .env                      # Actual secrets — NOT committed (listed in .gitignore)
│
├── agent/                    # Core back-end: agentic loop, tools, prompts
│   ├── __init__.py           # Exports CSRAllocationAgent for external use
│   ├── llm_agent.py          # Agentic loop: GPT-4o function calling + RAG injection
│   ├── tools.py              # Four tool functions + OpenAI JSON schema definitions
│   └── prompts.py            # System prompt, few-shot example, RAG query templates
│
├── rag/                      # Retrieval-Augmented Generation components
│   ├── __init__.py           # Exports RAGRetriever
│   ├── corpus_loader.py      # PDF text extraction, chunking, preprocessing
│   └── retriever.py          # FAISS index build, pickle cache, vector search
│
├── data/                     # Local data files (no download needed)
│   ├── SocialVulnerabilityIndex_LA.csv   # CDC SVI 2022 — Louisiana county-level
│   └── corpus/                           # Methodology PDFs for RAG + cached index
│       ├── fema_national-risk-index_technical-documentation.pdf
│       ├── SVI2022Documentation.pdf
│       ├── cejst-technical-support-document.pdf
│       └── .rag_index_cache.pkl          # Pre-built FAISS index (auto-regenerated if stale)
│
└── evaluation/               # Offline evaluation against historical events
    ├── __init__.py
    └── eval_runner.py        # Precision@5 metric + narrative quality rubric
```

---

## Attributions and Citations

The following external sources were used in building or grounding this project:

- **OpenAI API documentation** — Function calling patterns and best practices used in `agent/llm_agent.py` and `agent/tools.py`. https://platform.openai.com/docs/guides/function-calling
- **FAISS (Facebook AI Research)** — Efficient similarity search library used in `rag/retriever.py`. https://github.com/facebookresearch/faiss
- **Streamlit documentation** — UI component patterns used in `app.py`. https://docs.streamlit.io
- **FEMA National Risk Index Technical Documentation (2023)** — Source of NRI score methodology and rating thresholds. Included in `data/corpus/`.
- **CDC/ATSDR Social Vulnerability Index Documentation (2022)** — SVI theme structure and percentile interpretation. Included in `data/corpus/`.
- **Council on Environmental Quality, CEJST Technical Support Document (2022)** — Community burden indicators. Included in `data/corpus/`.
- **OpenFEMA API** — Disaster Declarations Summaries endpoint used in `tools.py`. https://www.fema.gov/about/openfema/api
- **U.S. Census Bureau ACS 5-Year Estimates API** — Socioeconomic variables used in `tools.py`. https://www.census.gov/data/developers/data-sets/acs-5year.html
- Cutter, S. L., Boruff, B. J., & Shirley, W. L. (2003). Social vulnerability to environmental hazards. *Social Science Quarterly, 84*(2), 242–261. — Social vulnerability framework foundation.
- Flanagan, B. E., Gregory, E. W., Hallisey, E. J., Heitgerd, J. L., & Lewis, B. (2011). A social vulnerability index for disaster management. *Journal of Homeland Security and Emergency Management, 8*(1). — CDC SVI methodology and emergency management application.
- Porter, M. E., & Kramer, M. R. (2006). Strategy and society: The link between competitive advantage and corporate social responsibility. *Harvard Business Review, 84*(12), 78–92. — Strategic CSR theory motivating the allocation framework.
