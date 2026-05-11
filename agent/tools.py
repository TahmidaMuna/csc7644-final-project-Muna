"""
tools.py
--------
Defines the four external tool functions available to the CSR allocation agent.
Each function calls a real public API and returns structured data for LLM reasoning.

Tools:
    - get_affected_parishes: OpenFEMA Disaster Declarations API
    - nri_lookup: FEMA National Risk Index ArcGIS REST API
    - svi_lookup: CDC Social Vulnerability Index ArcGIS REST API
    - census_lookup: U.S. Census Bureau ACS 5-Year Estimates API
"""

import csv
import json
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEMA_DISASTER_API = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
FEMA_NRI_API = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "National_Risk_Index_Census_Tracts/FeatureServer/0/query"
)
CDC_SVI_API = (
    "https://services1.arcgis.com/RLQu0rK7h4kbsBq5/arcgis/rest/services/"
    "CDC_Social_Vulnerability_Index_2022_USA_county/FeatureServer/0/query"
)
CENSUS_ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"
SVI_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "SocialVulnerabilityIndex_LA.csv"

# Louisiana state FIPS
LOUISIANA_FIPS = "22"

# OpenFEMA maps declaration types; we want all major disaster declarations
MAJOR_DISASTER_CODE = "DR"


def get_affected_parishes(disaster_number: Optional[int | str] = None, event: Optional[str] = None) -> dict:
    """
    Query the OpenFEMA Disaster Declarations API for a given disaster number
    and return all affected Louisiana parishes.

    Parameters
    ----------
    disaster_number : int or str
        FEMA disaster declaration number (e.g., 4611 for Hurricane Ida 2021).
        Values like "DR-4611" are accepted and normalized to 4611.
    event : str, optional
        Alternate argument name accepted for direct callers that pass event IDs
        like "DR-4611".

    Returns
    -------
    dict
        Keys: 'disaster_number', 'disaster_title', 'incident_type',
              'incident_begin_date', 'parishes' (list of dicts with
              'name' and 'fips').
    """
    raw_event = event if disaster_number is None else disaster_number
    if raw_event is None:
        return {"error": "Missing FEMA disaster number or event ID."}

    normalized_disaster_number = str(raw_event).upper().replace("DR-", "").strip()
    if not normalized_disaster_number.isdigit():
        return {"error": f"Invalid FEMA disaster number: {raw_event}"}

    params = {
        "$filter": (
            f"disasterNumber eq {normalized_disaster_number} "
            f"and state eq 'LA'"
        ),
        "$select": (
            "disasterNumber,state,declarationTitle,incidentType,"
            "incidentBeginDate,designatedArea,fipsCountyCode,fipsStateCode"
        ),
        "$format": "json",
        "$top": 100,
    }

    response = requests.get(FEMA_DISASTER_API, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    records = data.get("DisasterDeclarationsSummaries", [])
    if not records:
        return {
            "error": (
                "No FEMA records found for disaster number "
                f"{normalized_disaster_number} in Louisiana."
            )
        }

    # Deduplicate parishes by FIPS
    seen = set()
    parishes = []
    for rec in records:
        county_fips = rec.get("fipsCountyCode", "").zfill(3)
        full_fips = LOUISIANA_FIPS + county_fips
        name = rec.get("designatedArea", "Unknown")
        if full_fips not in seen:
            seen.add(full_fips)
            parishes.append({"name": name, "fips": full_fips})

    first = records[0]
    return {
        "disaster_number": int(normalized_disaster_number),
        "disaster_title": first.get("declarationTitle", "Unknown"),
        "incident_type": first.get("incidentType", "Unknown"),
        "incident_begin_date": first.get("incidentBeginDate", "Unknown"),
        "parishes": parishes,
    }


def nri_lookup(parish_fips: str) -> dict:
    """
    Retrieve FEMA National Risk Index scores for a Louisiana parish.

    The source service is tract-level, so this function aggregates all census
    tracts within the parish FIPS into a parish-level summary.

    Parameters
    ----------
    parish_fips : str
        Full 5-digit FIPS code for the parish (e.g., '22071' for Orleans).

    Returns
    -------
    dict
        NRI composite risk score, risk rating, hurricane expected annual loss,
        inland/coastal flood expected annual loss, and overall expected annual loss.
    """
    normalized_fips = str(parish_fips).strip().zfill(5)
    params = {
        "where": f"STCOFIPS = '{normalized_fips}'",
        "outFields": (
            "COUNTY,STCOFIPS,RISK_SCORE,RISK_RATNG,"
            "EAL_SCORE,EAL_RATNG,EAL_VALT,POPULATION,"
            "HWAV_EALS,HRCN_EALS,HRCN_EALT,TRND_EALS,"
            "IFLD_EALS,IFLD_EALT,CFLD_EALS,CFLD_EALT,"
            "SOVI_SCORE,SOVI_RATNG,RESL_SCORE,RESL_RATNG,CRF_VALUE"
        ),
        "f": "json",
        "returnGeometry": "false",
        "resultRecordCount": 2000,
    }

    response = requests.get(FEMA_NRI_API, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        details = "; ".join(data["error"].get("details", []))
        message = data["error"].get("message", "Unknown ArcGIS error")
        return {"error": f"NRI API error: {message}. {details}".strip()}

    features = data.get("features", [])
    if not features:
        return {"error": f"No NRI data found for FIPS {normalized_fips}."}

    records = [feature["attributes"] for feature in features]

    def safe_float(value: Optional[float | int | str]) -> Optional[float]:
        """
        Convert a raw attribute value to float, returning None for missing or invalid entries.

        Parameters
        ----------
        value : float, int, str, or None
            Raw value from an ArcGIS feature attribute.

        Returns
        -------
        float or None
        """
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def weighted_average(field: str) -> Optional[float]:
        """
        Compute a population-weighted average of a numeric field across all tracts.

        Falls back to an unweighted mean when all population values are zero.

        Parameters
        ----------
        field : str
            ArcGIS attribute name to aggregate.

        Returns
        -------
        float or None
            Rounded to 4 decimal places, or None if no valid values exist.
        """
        values = []
        for attrs in records:
            value = safe_float(attrs.get(field))
            if value is None:
                continue
            weight = safe_float(attrs.get("POPULATION")) or 0
            values.append((value, weight))

        if not values:
            return None

        total_weight = sum(weight for _, weight in values)
        if total_weight > 0:
            return round(sum(value * weight for value, weight in values) / total_weight, 4)

        # Unweighted fallback when population data is missing
        return round(sum(value for value, _ in values) / len(values), 4)

    def sum_field(field: str) -> Optional[float]:
        """
        Sum a numeric field across all tracts, used for dollar-value EAL aggregation.

        Parameters
        ----------
        field : str
            ArcGIS attribute name to sum.

        Returns
        -------
        float or None
            Rounded total, or None if no valid values exist.
        """
        values = [safe_float(attrs.get(field)) for attrs in records]
        values = [value for value in values if value is not None]
        return round(sum(values), 2) if values else None

    def weighted_rating(field: str) -> Optional[str]:
        """
        Determine the population-majority rating category for a categorical field.

        Selects the rating string (e.g., 'Very High', 'Moderate') whose tracts
        collectively contain the most population.

        Parameters
        ----------
        field : str
            ArcGIS attribute name holding a rating string.

        Returns
        -------
        str or None
            Most common rating weighted by population, or None if unavailable.
        """
        weights: dict[str, float] = {}
        for attrs in records:
            rating = attrs.get(field)
            if not rating:
                continue
            weight = safe_float(attrs.get("POPULATION")) or 1
            weights[rating] = weights.get(rating, 0) + weight
        if not weights:
            return None
        return max(weights.items(), key=lambda item: item[1])[0]

    first = records[0]
    county = first.get("COUNTY", "Unknown")
    parish_name = county if str(county).endswith("Parish") else f"{county} Parish"

    return {
        "parish": parish_name,
        "fips": normalized_fips,
        "tract_count": len(records),
        "risk_score": weighted_average("RISK_SCORE"),
        "risk_rating": weighted_rating("RISK_RATNG"),
        "expected_annual_loss_score": weighted_average("EAL_SCORE"),
        "expected_annual_loss_rating": weighted_rating("EAL_RATNG"),
        "expected_annual_loss_usd": sum_field("EAL_VALT"),
        "hurricane_eal_score": weighted_average("HRCN_EALS"),
        "hurricane_eal_usd": sum_field("HRCN_EALT"),
        "inland_flood_eal_score": weighted_average("IFLD_EALS"),
        "inland_flood_eal_usd": sum_field("IFLD_EALT"),
        "coastal_flood_eal_score": weighted_average("CFLD_EALS"),
        "coastal_flood_eal_usd": sum_field("CFLD_EALT"),
        "social_vulnerability_score": weighted_average("SOVI_SCORE"),
        "social_vulnerability_rating": weighted_rating("SOVI_RATNG"),
        "community_resilience_score": weighted_average("RESL_SCORE"),
        "community_resilience_rating": weighted_rating("RESL_RATNG"),
        "source": "FEMA National Risk Index Census Tracts FeatureServer",
        "aggregation_method": "Population-weighted tract averages for scores; tract sums for dollar EAL values.",
    }


def svi_lookup(parish_fips: str) -> dict:
    """
    Retrieve CDC Social Vulnerability Index (2022) scores for a parish
    from the local Louisiana SVI CSV.

    Parameters
    ----------
    parish_fips : str
        Full 5-digit FIPS code (e.g., '22071').

    Returns
    -------
    dict
        Overall SVI percentile and four theme scores:
        socioeconomic status, household characteristics,
        racial/ethnic minority status, housing type/transportation.
    """
    normalized_fips = str(parish_fips).strip().zfill(5)
    if not SVI_CSV_PATH.exists():
        return {"error": f"SVI CSV not found at {SVI_CSV_PATH}."}

    def safe_float(value: Optional[str]) -> Optional[float]:
        """
        Convert a CSV string value to float, returning None for blank or non-numeric entries.

        Parameters
        ----------
        value : str or None
            Raw string from the SVI CSV column.

        Returns
        -------
        float or None
        """
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    with SVI_CSV_PATH.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        attrs = next((row for row in reader if row.get("FIPS") == normalized_fips), None)

    if not attrs:
        return {"error": f"No SVI data found for FIPS {normalized_fips}."}

    return {
        "parish": attrs.get("COUNTY", "Unknown"),
        "fips": normalized_fips,
        "overall_svi_percentile": safe_float(attrs.get("RPL_THEMES")),
        "theme1_socioeconomic": safe_float(attrs.get("RPL_THEME1")),
        "theme2_household_characteristics": safe_float(attrs.get("RPL_THEME2")),
        "theme3_racial_ethnic_minority": safe_float(attrs.get("RPL_THEME3")),
        "theme4_housing_transportation": safe_float(attrs.get("RPL_THEME4")),
        "total_population": safe_float(attrs.get("E_TOTPOP")),
        "persons_below_150pct_poverty": safe_float(attrs.get("E_POV150")),
        "unemployed": safe_float(attrs.get("E_UNEMP")),
        "uninsured": safe_float(attrs.get("E_UNINSUR")),
        "minority_population": safe_float(attrs.get("E_MINRTY")),
    }


def census_lookup(parish_fips: str, census_api_key: Optional[str] = None) -> dict:
    """
    Retrieve ACS 5-Year socioeconomic estimates for a Louisiana parish.

    Parameters
    ----------
    parish_fips : str
        Full 5-digit FIPS code (e.g., '22071').
    census_api_key : str, optional
        U.S. Census Bureau API key. If omitted, unauthenticated requests are used
        (rate-limited but functional for development).

    Returns
    -------
    dict
        Median household income, poverty rate, uninsured rate, and housing cost burden.
    """
    county_fips = parish_fips[2:]  # last 3 digits
    variables = ",".join([
        "B19013_001E",   # Median household income
        "B17001_002E",   # Persons in poverty
        "B17001_001E",   # Total for poverty universe
        "B27010_017E",   # Uninsured under 19
        "B27010_033E",   # Uninsured 19-64
        "B25070_010E",   # Housing cost burden >= 50%
        "B25070_001E",   # Total renters
        "B01003_001E",   # Total population
    ])

    params = {
        "get": f"NAME,{variables}",
        "for": f"county:{county_fips}",
        "in": f"state:{LOUISIANA_FIPS}",
    }
    if census_api_key:
        params["key"] = census_api_key

    response = requests.get(CENSUS_ACS_BASE, params=params, timeout=15)
    response.raise_for_status()
    rows = response.json()

    if len(rows) < 2:
        return {"error": f"No Census ACS data found for FIPS {parish_fips}."}

    header, values = rows[0], rows[1]
    data = dict(zip(header, values))

    def safe_float(key: str) -> Optional[float]:
        """Convert Census string value to float, returning None for negatives."""
        val = data.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return f if f >= 0 else None
        except (ValueError, TypeError):
            return None

    median_income = safe_float("B19013_001E")
    in_poverty = safe_float("B17001_002E")
    poverty_universe = safe_float("B17001_001E")
    uninsured_under19 = safe_float("B27010_017E") or 0
    uninsured_19_64 = safe_float("B27010_033E") or 0
    housing_burdened = safe_float("B25070_010E")
    total_renters = safe_float("B25070_001E")
    total_pop = safe_float("B01003_001E")

    poverty_rate = (in_poverty / poverty_universe * 100) if poverty_universe else None
    housing_burden_rate = (housing_burdened / total_renters * 100) if total_renters else None
    uninsured_count = uninsured_under19 + uninsured_19_64

    return {
        "parish": data.get("NAME", "Unknown"),
        "fips": parish_fips,
        "median_household_income_usd": median_income,
        "poverty_rate_pct": round(poverty_rate, 2) if poverty_rate else None,
        "uninsured_count": uninsured_count,
        "housing_cost_burden_pct": round(housing_burden_rate, 2) if housing_burden_rate else None,
        "total_population": total_pop,
    }


# ---------------------------------------------------------------------------
# JSON schema definitions for OpenAI function calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_affected_parishes",
            "description": (
                "Query the OpenFEMA Disaster Declarations API to retrieve all Louisiana "
                "parishes affected by a given federal disaster declaration number, along "
                "with disaster metadata (title, incident type, begin date)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disaster_number": {
                        "type": "integer",
                        "description": (
                            "FEMA federal disaster declaration number "
                            "(e.g., 4611 for Hurricane Ida 2021, 4277 for August 2016 floods)."
                        ),
                    }
                },
                "required": ["disaster_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nri_lookup",
            "description": (
                "Retrieve FEMA National Risk Index composite risk score, expected annual loss, "
                "hurricane EAL, and flood EAL for a Louisiana parish by its 5-digit FIPS code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parish_fips": {
                        "type": "string",
                        "description": "5-digit FIPS code for the parish (e.g., '22071').",
                    }
                },
                "required": ["parish_fips"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "svi_lookup",
            "description": (
                "Retrieve CDC Social Vulnerability Index 2022 scores for a Louisiana parish "
                "by FIPS code. Returns overall SVI percentile and four theme scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parish_fips": {
                        "type": "string",
                        "description": "5-digit FIPS code for the parish (e.g., '22071').",
                    }
                },
                "required": ["parish_fips"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "census_lookup",
            "description": (
                "Retrieve ACS 5-Year socioeconomic estimates for a Louisiana parish: "
                "median household income, poverty rate, uninsured count, and housing cost burden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parish_fips": {
                        "type": "string",
                        "description": "5-digit FIPS code for the parish (e.g., '22071').",
                    }
                },
                "required": ["parish_fips"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatcher: maps tool name strings to Python callables
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "get_affected_parishes": get_affected_parishes,
    "nri_lookup": nri_lookup,
    "svi_lookup": svi_lookup,
    "census_lookup": census_lookup,
}


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool by name with the given arguments and return the result as JSON.

    Parameters
    ----------
    tool_name : str
        Name of the tool function to call.
    tool_args : dict
        Arguments parsed from the LLM's function call request.

    Returns
    -------
    str
        JSON string of the tool's return value, for injection back into the LLM context.
    """
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = fn(**tool_args)
        return json.dumps(result, default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
