"""
app.py
------
Streamlit web application for the Post-Disaster CSR Fund Allocation Agent.

Provides a browser-based interface for CSR analysts to:
    - Select a preset FEMA disaster event or enter a custom disaster number
    - Trigger the agentic analysis pipeline (GPT-4o + live federal APIs)
    - View the structured parish allocation report rendered as Markdown
    - Download the report as a plain-text file

Environment variables (loaded from .env):
    OPENAI_API_KEY : Required. Used to authenticate the LLM and embedding calls.
    CENSUS_API_KEY : Optional. Increases Census ACS rate limits.

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from agent.llm_agent import CSRAllocationAgent

load_dotenv(".env")

# Read API key at module load time; absence is caught at run-button click rather than startup
# so the UI still renders for users who want to explore without immediately running the agent.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
CORPUS_DIR = "data/corpus"
VERBOSE_MODE = False


# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CSR Allocation Agent | Louisiana Disaster Relief",
    page_icon="🌊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: Documentation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown(
        """
        This tool helps Louisiana oil and gas companies make **data-driven CSR allocation decisions** 
        after a disaster event.
        
        The agent automatically:
        1. Identifies affected parishes via **OpenFEMA API**
        2. Retrieves hazard severity from **FEMA NRI**
        3. Retrieves social vulnerability from **CDC SVI**
        4. Retrieves socioeconomic data from **U.S. Census ACS**
        5. Ranks parishes by weighted need score
        6. Generates a cited allocation report

        ### Need Score Methodology
        The composite need score combines four normalized indicators:

        - **NRI Risk Score:** 30% — hazard severity and expected losses
        - **SVI Overall Percentile:** 40% — social vulnerability
        - **Poverty Rate:** 20% — economic hardship
        - **Housing Cost Burden:** 10% — housing fragility
        
        ---
        *CSC 7644: Applied LLM Development*  
        *Tahmida Sarker Muna | LSU*
        """
    )

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.title("🌊 Post-Disaster CSR Fund Allocation Agent")
st.caption(
    "Louisiana Oil & Gas | Corporate Social Responsibility | Powered by GPT-4o + FEMA/CDC/Census APIs"
)

st.info(
    "Select a preset FEMA disaster event and review the query below. "
    "The agent will retrieve live data from public APIs and generate a prioritized allocation report."
)

# ---------------------------------------------------------------------------
# Disaster event selector
# ---------------------------------------------------------------------------

# Maps human-readable labels to their numeric FEMA declaration numbers.
# These four events have well-documented parish-level damage records for Louisiana.
PRESET_MAP = {
    "Hurricane Ida 2021 (DR-4611)": 4611,
    "August 2016 Louisiana Floods (DR-4277)": 4277,
    "Hurricane Laura 2020 (DR-4559)": 4559,
    "Hurricane Katrina 2005 (DR-1603)": 1603,
}

col1, col2 = st.columns([1, 2])

with col1:
    preset = st.selectbox(
        "Select a preset disaster event",
        options=list(PRESET_MAP.keys()),
    )

with col2:
    disaster_number = PRESET_MAP[preset]
    st.metric("Disaster Number", f"DR-{disaster_number}")

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

user_query = st.text_area(
    "Describe your CSR allocation goal",
    value=(
        f"Our company wants to direct CSR funds to communities most affected by disaster "
        f"number {disaster_number} in Louisiana. Please retrieve hazard, vulnerability, "
        f"and socioeconomic data for all affected parishes and generate a prioritized "
        f"allocation recommendation."
    ),
    height=120,
)

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_button:
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY is missing. Add it to the project .env file and restart the app.")
        st.stop()

    if not user_query.strip():
        st.warning("Please enter a query before running.")
        st.stop()

    # Initialize agent — raises ValueError if API key is missing or empty
    try:
        agent = CSRAllocationAgent(
            openai_api_key=OPENAI_API_KEY,
            corpus_dir=CORPUS_DIR,
            verbose=VERBOSE_MODE,
        )
    except ValueError as e:
        st.error(f"❌ Agent initialization failed: {e}")
        st.stop()

    # Run the agentic loop; this typically takes 30–60 s due to multiple API calls
    with st.spinner("🔍 Agent is retrieving data and analyzing parishes... (may take 30–60s)"):
        progress_placeholder = st.empty()
        start_time = time.time()

        try:
            report = agent.run(user_query)
            elapsed = round(time.time() - start_time, 1)
            progress_placeholder.empty()  # Clear the placeholder once results are ready
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Agent error: {e}")
            st.stop()

    st.success(f"✅ Analysis complete in {elapsed}s")

    # ---------------------------------------------------------------------------
    # Display the report
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📋 CSR Allocation Report")

    # Render as Markdown so that the agent's headers, bold text, and
    # bullet lists display correctly in the browser.
    st.markdown(report)

    # ---------------------------------------------------------------------------
    # Download the report
    # ---------------------------------------------------------------------------
    st.divider()

    # Timestamp the filename so repeated runs for the same disaster don't overwrite each other
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"csr_allocation_DR{disaster_number}_{timestamp}.txt"

    st.download_button(
        label="📥 Download Report (.txt)",
        data=report,
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Disaster: DR-{disaster_number} | Latency: {elapsed}s"
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data sources: OpenFEMA Disaster Declarations API · FEMA National Risk Index (ArcGIS REST) · "
    "CDC Social Vulnerability Index 2022 (local CSV) · U.S. Census Bureau ACS 5-Year Estimates · "
    "All sources are free and publicly accessible."
)
