# ============================================================
# FULL TCO STREAMLIT APP
# Full TCO Model version:
#   1) No separate MDPI/Interactive modes in Streamlit
#   2) Uses four separate run buttons so only one application is calculated/displayed at a time
#   3) CNG is included for Refuse, Transit Bus, Drayage, and Long Haul
#   4) Drayage/Long Haul CNG are trial values copied from diesel inputs
#   5) Sidebar inputs are grouped by selected application and vehicle type
#   6) Breakeven uses exact diesel LCOD from same mother LCOD run
#   7) Adds optional federal corporate tax-shield calculations and
#      separate pre-tax / after-tax Full TCO plots
#   8) Adds absolute total PV TCO plots and restricts ton-mile outputs
#      to freight applications (Drayage and Long Haul)
#   9) Adds two Full TCO mileage modes:
#      constant annual mileage and approximate Argonne/VIUS age-dependent VMT
#  10) Reduces Streamlit load by running one selected case at a time
# ============================================================

import copy
import random
from typing import Dict, Tuple, Any, Literal
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Full TCO Model", layout="wide")

st.title("Full TCO Model")
st.markdown(
    "Argonne-style discounted Full TCO model for Refuse, Transit Bus, Drayage, and Long Haul. "
    "Existing LCOD app inputs are preserved; added economic, mileage-mode, and federal corporate tax-shield inputs are managed separately. Use the four case buttons to calculate and display one application at a time. "
    "The optional Argonne/VIUS mileage mode uses approximate age-dependent mileage factors digitized from Argonne Figure 2.7 and scaled to the user-entered average annual mileage."
)

# ============================================================
# PLOT STYLE
# ============================================================
FONT_XTICK = 14
FONT_YTICK = 14
FONT_AXES_LABEL = 14
FONT_AXES_TITLE = 15
FONT_LEGEND = 12
FONT_FIGURE = 18

plt.rcParams.update({
    "xtick.labelsize": FONT_XTICK,
    "ytick.labelsize": FONT_YTICK,
    "axes.labelsize": FONT_AXES_LABEL,
    "axes.titlesize": FONT_AXES_TITLE,
    "legend.fontsize": FONT_LEGEND,
    "figure.titlesize": FONT_FIGURE,
})

# ============================================================
# TYPES AND CONSTANTS
# ============================================================
Scenario = Literal["current", "2030"]
Range = Tuple[float, float]

DEFAULT_N_SAMPLES = 20_000
DEFAULT_RANDOM_SEED = 7
PCTILES = (5, 50, 95)

APP_ORDER = ["refuse", "bus", "drayage", "longhaul"]
FREIGHT_TON_MILE_APPS = ["drayage", "longhaul"]
VEHICLE_ORDER = ["diesel", "fcev", "bev", "cng"]

# Mileage modes used only for the discounted Full TCO calculations.
# The original LCOD/breakeven calculation is intentionally kept unchanged.
MILEAGE_MODE_CONSTANT = "Constant annual mileage"
MILEAGE_MODE_VIUS = "Argonne/VIUS approximate age-dependent mileage"
MILEAGE_MODE_OPTIONS = [MILEAGE_MODE_CONSTANT, MILEAGE_MODE_VIUS]


def _format_dollar_axis_millions(value, pos=None):
    """Format large dollar axis values without Matplotlib scientific notation."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{sign}${abs_value / 1_000:.0f}k"
    return f"{sign}${abs_value:.0f}"

# Approximate year-by-year MHDV VMT schedules visually digitized from
# Argonne ANL/ESD-21/4 Figure 2.7.  These are used as normalized shapes,
# not as fixed absolute mileages: the code scales the curve so the user's
# selected average annual mileage is preserved over the sampled lifetime.
ARGONNE_VIUS_MHDV_VMT = {
    # Our app's Long Haul maps to Argonne Tractor - Sleeper Cab.
    "longhaul": [
        108000.0, 120000.0, 114000.0, 105000.0, 92000.0,
        80000.0, 72000.0, 64000.0, 56000.0, 50000.0,
        44000.0, 39000.0, 35000.0, 31000.0, 27000.0,
    ],
    # Our app's Drayage maps to Argonne Tractor - Day Cab.
    "drayage": [
        74000.0, 73000.0, 72000.0, 68000.0, 60000.0,
        53000.0, 47000.0, 42000.0, 38000.0, 34000.0,
        30000.0, 27000.0, 24500.0, 22500.0, 20500.0,
    ],
    # Our app's Refuse maps to Argonne Class 8 Refuse.
    "refuse": [
        30000.0, 31000.0, 31000.0, 30000.0, 28000.0,
        26500.0, 25000.0, 24000.0, 23500.0, 22500.0,
        21000.0, 19500.0, 17500.0, 16000.0, 15000.0,
    ],
    # Our app's Transit Bus maps to Argonne Transit Bus.
    "bus": [
        23000.0, 38000.0, 42000.0, 42000.0, 41000.0,
        39500.0, 39000.0, 39000.0, 39000.0, 38500.0,
        38000.0, 38000.0, 37500.0, 37000.0, 36500.0,
    ],
}

ARGONNE_VIUS_LABELS = {
    "longhaul": "Argonne Tractor - Sleeper Cab",
    "drayage": "Argonne Tractor - Day Cab",
    "refuse": "Argonne Class 8 Refuse",
    "bus": "Argonne Transit Bus",
}

VEHICLE_COLORS = {
    "diesel": "#4D4D4D",
    "fcev": "#1F77B4",
    "bev": "#2CA02C",
    "cng": "#FF7F0E",
}



# Full TCO component order for stacked plots
FULL_TCO_COMPONENTS = [
    "vehicle",
    "financing",
    "fuel",
    "maintenance",
    "insurance",
    "tax_fees",
    "payload",
    "labor",
]

AFTER_TAX_COMPONENTS = FULL_TCO_COMPONENTS + ["federal_tax_benefit"]

FULL_TCO_LABELS = {
    "vehicle": "Vehicle",
    "financing": "Financing",
    "fuel": "Fuel",
    "maintenance": "Maintenance",
    "insurance": "Insurance",
    "tax_fees": "Taxes & fees",
    "payload": "Payload",
    "labor": "Labor",
    "federal_tax_benefit": "Federal tax benefit",
}

# Explicit colors for component-stacked Full TCO bars
FULL_TCO_COLORS = {
    "vehicle": "#FF6B6B",      # bright coral red
    "financing": "#9B5DE5",    # bright purple
    "fuel": "#00B0F0",         # bright blue
    "maintenance": "#00CC66",  # bright green
    "insurance": "#00D5D5",    # bright cyan
    "tax_fees": "#FFA600",     # bright orange
    "payload": "#FFD166",      # bright yellow
    "labor": "#FF3D8B",        # bright pink
    "federal_tax_benefit": "#7F7F7F",  # gray tax-shield reduction
}

ECON_MODE_DEFAULT = "Use Argonne low-advancement default values"
ECON_MODE_CUSTOM = "Use custom values for this application"
ECON_MODE_OPTIONS = [ECON_MODE_DEFAULT, ECON_MODE_CUSTOM]

REQUIRED_RES_KEYS = [
    "initial_cost_current",
    "initial_cost_2030",
    "residual_factor_current",
    "residual_factor_2030",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def _is_range(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)

def _sample_uniform(rng: random.Random, r: Range) -> float:
    lo, hi = r
    if lo > hi:
        raise ValueError(f"Bad range {r}: lo > hi")
    return lo if lo == hi else lo + (hi - lo) * rng.random()

def _summarize_percentiles(arr: np.ndarray, pctiles=(5, 50, 95)) -> Dict[str, float]:
    return {f"p{p}": float(np.percentile(arr, p)) for p in pctiles}

def _as_range(x: Range | float | int) -> Range:
    if isinstance(x, (int, float)):
        return (float(x), float(x))
    return (float(x[0]), float(x[1]))

def _cost_range(power_kw: Range | float, price_per_kw: Range | float, max_mult: float = 1.0) -> Range:
    p_lo, p_hi = _as_range(power_kw)
    c_lo, c_hi = _as_range(price_per_kw)
    return (float(p_lo * c_lo), float(p_hi * c_hi * float(max_mult)))

def _h2_storage_cost_range(h2_storage_required: Range | float, max_mult: float = 1.0) -> Range:
    s_lo, s_hi = _as_range(h2_storage_required)
    k = 12.7 * 33.1
    return (float(k * s_lo), float(k * s_hi * float(max_mult)))

def _vehicle_pretty(vt: str) -> str:
    return vt.upper()

def bev_battery_mass_kg(battery_energy_kwh: float, energy_density_kwh_per_kg: float) -> float:
    if energy_density_kwh_per_kg <= 0:
        raise ValueError("BEV energy density must be > 0.")
    return battery_energy_kwh / energy_density_kwh_per_kg

def bev_revenue_weight_ton(rev_weight_ton_diesel_fcev: float, battery_mass_kg: float) -> float:
    return rev_weight_ton_diesel_fcev - battery_mass_kg / 1000.0

def residual_cost_usd_point(residual_tbl: Dict[str, Any], vehicle_type: str, scenario: Scenario) -> float:
    comps = residual_tbl[vehicle_type]
    tot = 0.0
    for comp in comps.values():
        if scenario == "current":
            tot += comp["initial_cost_current"] * comp["residual_factor_current"]
        else:
            tot += comp["initial_cost_2030"] * comp["residual_factor_2030"]
    return float(tot)

def lcod_usd_per_ton_mile_point(lcod_mile: float, revenue_weight_ton: float) -> float:
    if revenue_weight_ton <= 0:
        raise ValueError("Revenue weight must be > 0.")
    return float(lcod_mile / revenue_weight_ton)

def range_input(label: str, default_range: Range, key: str) -> Range:
    col1, col2 = st.columns(2)
    with col1:
        lo = st.number_input(f"{label} min", value=float(default_range[0]), key=f"{key}_lo")
    with col2:
        hi = st.number_input(f"{label} max", value=float(default_range[1]), key=f"{key}_hi")
    return (lo, hi)


def percent_range_input(label: str, default_range: Range, key: str) -> Range:
    """Show a fraction range as percent in the sidebar and return fractions."""
    col1, col2 = st.columns(2)
    with col1:
        lo_pct = st.number_input(
            f"{label} min (%)",
            value=float(default_range[0]) * 100.0,
            key=f"{key}_lo_pct",
            step=0.1,
            format="%.4f",
        )
    with col2:
        hi_pct = st.number_input(
            f"{label} max (%)",
            value=float(default_range[1]) * 100.0,
            key=f"{key}_hi_pct",
            step=0.1,
            format="%.4f",
        )
    return (lo_pct / 100.0, hi_pct / 100.0)


def smart_econ_range_input(label: str, default_range: Range, key: str) -> Range:
    """Use percent widgets for rates/fractions, normal range widgets otherwise."""
    percent_keys = [
        "discount_rate",
        "loan_apr",
        "down_payment_fraction",
        "sales_tax_fraction",
        "federal_excise_tax_fraction",
        "corporate_tax_rate",
        "tax_benefit_utilization_fraction",
    ]
    if label in percent_keys:
        return percent_range_input(label, default_range, key)
    return range_input(label, default_range, key)


def econ_input_set_label(mode: str) -> str:
    return "Customized" if mode == ECON_MODE_CUSTOM else "Default"


# ============================================================
# ECONOMIC INPUT DISPLAY HELPERS
# ============================================================
def economic_input_unit(key: str) -> str:
    """Human-readable unit labels for default/custom economic inputs."""
    percent_keys = {
        "discount_rate",
        "loan_apr",
        "down_payment_fraction",
        "sales_tax_fraction",
        "federal_excise_tax_fraction",
        "corporate_tax_rate",
        "tax_benefit_utilization_fraction",
    }
    if key in percent_keys:
        return "%"
    if key == "loan_term_years":
        return "years"
    if key == "hvut_weight_rating_lb" or key == "empty_weight_lb":
        return "lb"
    if "usd_per_mile" in key:
        return "$/mile"
    if "usd_per_year" in key:
        return "$/year"
    if "usd_per_1000_month" in key:
        return "$/month per $1,000 vehicle value"
    if key.endswith("_usd"):
        return "$"
    if key == "hvut_exempt_flag":
        return "0 = not exempt, 1 = exempt"
    return ""


def economic_input_display_name(key: str) -> str:
    """Cleaner display labels for the economic input tables."""
    label_map = {
        "discount_rate": "Discount rate",
        "loan_apr": "Loan APR",
        "down_payment_fraction": "Down payment",
        "loan_term_years": "Loan term",
        "insurance_fixed_usd_per_year": "Fixed insurance",
        "insurance_liability_usd_per_mile": "Liability insurance",
        "insurance_physical_damage_usd_per_1000_month": "Physical damage insurance",
        "sales_tax_fraction": "Sales tax",
        "federal_excise_tax_fraction": "Federal excise tax",
        "initial_registration_usd": "Initial registration",
        "documentation_fee_usd": "Documentation fee",
        "annual_registration_usd_per_year": "Annual registration",
        "registration_weight_rate_usd_per_lb_year": "Weight-based registration rate",
        "empty_weight_lb": "Empty weight",
        "hvut_weight_rating_lb": "HVUT weight rating",
        "hvut_exempt_flag": "HVUT exemption flag",
        "permits_licenses_tolls_usd_per_mile": "Permits/licenses/tolls",
        "other_annual_fees_usd_per_year": "Other annual fees",
        "driver_labor_usd_per_mile": "Driver labor",
        "payload_penalty_usd_per_mile": "Payload penalty",
        "fueling_or_charging_labor_usd_per_mile": "Fueling/charging labor",
        "annual_afv_registration_usd_per_year": "Annual AFV registration surcharge",
        "corporate_tax_rate": "Federal corporate tax rate",
        "tax_benefit_utilization_fraction": "Tax benefit utilization",
    }
    return label_map.get(key, key.replace("_", " ").title())


def format_econ_value(key: str, value: float) -> str:
    """Format values for the read-only default/current economic tables."""
    percent_keys = {
        "discount_rate",
        "loan_apr",
        "down_payment_fraction",
        "sales_tax_fraction",
        "federal_excise_tax_fraction",
        "corporate_tax_rate",
        "tax_benefit_utilization_fraction",
    }
    if key in percent_keys:
        return f"{float(value) * 100.0:.4g}%"
    if key == "hvut_exempt_flag":
        return "Exempt" if float(value) >= 0.5 else "Not exempt"
    if abs(float(value)) >= 1000:
        return f"{float(value):,.2f}"
    return f"{float(value):.4g}"


def economic_tables_from_config(econ_cfg: Dict[str, Any]):
    """Return application-level and vehicle-level economic input tables."""
    app_rows = []
    for key, rng in econ_cfg["APP_R"].items():
        app_rows.append({
            "Input": economic_input_display_name(key),
            "Min": format_econ_value(key, rng[0]),
            "Max": format_econ_value(key, rng[1]),
            "Unit": economic_input_unit(key),
        })

    veh_rows = []
    for vt in VEHICLE_ORDER:
        if vt not in econ_cfg["VEH_R"]:
            continue
        for key, rng in econ_cfg["VEH_R"][vt].items():
            veh_rows.append({
                "Vehicle": vt.upper(),
                "Input": economic_input_display_name(key),
                "Min": format_econ_value(key, rng[0]),
                "Max": format_econ_value(key, rng[1]),
                "Unit": economic_input_unit(key),
            })

    return pd.DataFrame(app_rows), pd.DataFrame(veh_rows)


def show_economic_values_table(econ_cfg: Dict[str, Any], title: str):
    """Display read-only economic input values in the sidebar."""
    st.markdown(f"**{title}**")
    app_df, veh_df = economic_tables_from_config(econ_cfg)

    st.caption("Application-level inputs")
    st.dataframe(app_df, hide_index=True, width="stretch", height=260)

    st.caption("Vehicle-specific inputs")
    st.dataframe(veh_df, hide_index=True, width="stretch", height=260)

# ============================================================
# BASE APPLICATION DATA
# ============================================================
def build_base_applications() -> Dict[str, Any]:

    applications = {
        # --------------------------------------------------------
        # REFUSE
        # --------------------------------------------------------
        "refuse": {
            "label": "Refuse",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (300.0, 310.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (319000.0, 355000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.8),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (445000.0, 540000.0),
                    "fuel_economy_mi_per_unit": (5.0, 6.0),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (438575.0, 671000.0),
                    "fuel_economy_mi_per_unit": (0.30, 0.32),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "cng": {
                    "purchase_cost_usd": (406000.0, 485000.0),
                    "fuel_economy_mi_per_unit": (1.90, 2.40),
                    "fuel_price_usd_per_unit": (2.90, 3.20),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.72, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # TRANSIT BUS
        # --------------------------------------------------------
        "bus": {
            "label": "Transit Bus",
            "price_mode": "base_plus_premium",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (300.0, 310.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "base_price_usd": (440000.0, 450000.0),
                    "premium_price_usd": (0.0, 0.0),
                    "fuel_economy_mi_per_unit": (3.0, 4.0),
                    "fuel_price_usd_per_unit": (3.5, 4.0),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "base_price_usd": (385455.0, 411438.0),
                    "premium_price_usd": (350000.0, 738562.0),
                    "fuel_economy_mi_per_unit": (6.5, 7.5),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "base_price_usd": (385000.0, 514585.0),
                    "premium_price_usd": (350000.0, 585415.0),
                    "fuel_economy_mi_per_unit": (0.34, 0.40),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "cng": {
                    "base_price_usd": (600000.0, 650000.0),
                    "premium_price_usd": (0.0, 0.0),
                    "fuel_economy_mi_per_unit": (2.80, 3.00),
                    "fuel_price_usd_per_unit": (2.90, 3.20),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.72, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # DRAYAGE
        # --------------------------------------------------------
        "drayage": {
            "label": "Drayage",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (180.0, 378.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (151000.0, 178000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.2),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (348000.0, 385000.0),
                    "fuel_economy_mi_per_unit": (5.1, 6.9),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (233575.0, 388000.0),
                    "fuel_economy_mi_per_unit": (0.27, 0.30),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                # Trial CNG values copied from diesel for first-pass comparison.
                # Replace these with literature CNG parameters when finalized.
                "cng": {
                    "purchase_cost_usd": (151000.0, 178000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.2),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # LONG HAUL
        # --------------------------------------------------------
        "longhaul": {
            "label": "Long Haul",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (650.0, 850.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (171000.0, 210000.0),
                    "fuel_economy_mi_per_unit": (6.0, 7.0),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (430000.0, 489000.0),
                    "fuel_economy_mi_per_unit": (7.0, 8.1),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (410575.0, 458000.0),
                    "fuel_economy_mi_per_unit": (0.43, 0.59),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                # Trial CNG values copied from diesel for first-pass comparison.
                # Replace these with literature CNG parameters when finalized.
                "cng": {
                    "purchase_cost_usd": (171000.0, 210000.0),
                    "fuel_economy_mi_per_unit": (6.0, 7.0),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
            },
        },
    }

    # Drayage and Long Haul CNG are already included above as diesel-like trial values.

    return applications

# ============================================================
# ADD RESIDUAL TABLES
# ============================================================
def add_residual_tables(applications: Dict[str, Any]) -> Dict[str, Any]:

    applications["refuse"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(237000.0, 237000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((200.0, 250.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((80.0, 100.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((130.0, 130.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((25.0, 25.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["bus"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(293000.0, 300000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((384.0, 400.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((80.0, 80.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((85.0, 100.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((60.0, 70.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(400000.0, 433000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["drayage"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(119000.0, 119000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((180.0, 378.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((70.0, 160.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((168.0, 210.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((26.0, 40.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["longhaul"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(134000.0, 134000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((650.0, 850.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((200.0, 300.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((318.0, 350.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((60.0, 75.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    return applications

# ============================================================
# VEHICLE COMPLETENESS CHECK
# ============================================================
def vehicle_is_complete(app_cfg: Dict[str, Any], vt: str) -> bool:
    VEH_R = app_cfg["VEH_R"]
    RESIDUAL_R = app_cfg["RESIDUAL_R"]
    price_mode = app_cfg["price_mode"]

    if vt not in VEH_R or vt not in RESIDUAL_R:
        return False

    v = VEH_R[vt]
    if not isinstance(v, dict):
        return False

    if price_mode == "single_purchase":
        req_keys = [
            "purchase_cost_usd",
            "fuel_economy_mi_per_unit",
            "fuel_price_usd_per_unit",
            "planned_miles_per_year",
            "maintenance_usd_per_mile",
        ]
    else:
        req_keys = [
            "base_price_usd",
            "premium_price_usd",
            "fuel_economy_mi_per_unit",
            "fuel_price_usd_per_unit",
            "planned_miles_per_year",
            "maintenance_usd_per_mile",
        ]

    for k in req_keys:
        if k not in v or v[k] is None or not _is_range(v[k]):
            return False

    for _, fields in RESIDUAL_R[vt].items():
        for rk in REQUIRED_RES_KEYS:
            if rk not in fields or fields[rk] is None or not _is_range(fields[rk]):
                return False

    return True

# ============================================================
# LCOD COMPONENT FUNCTIONS
# ============================================================
def lcod_components_per_mile_single_purchase(v, g, residual_cost):
    fe = v["fuel_economy_mi_per_unit"]
    if fe <= 0:
        raise ValueError("Fuel economy must be > 0.")

    lifetime_miles = g["lifetime_years"] * v["planned_miles_per_year"]
    if lifetime_miles <= 0:
        raise ValueError("Lifetime miles must be > 0.")

    fuel_per_mile = (1.0 / fe) * v["fuel_price_usd_per_unit"]
    purchase_per_mile = (v["purchase_cost_usd"] - residual_cost) / lifetime_miles
    maintenance_per_mile = v["maintenance_usd_per_mile"]
    total = fuel_per_mile + purchase_per_mile + maintenance_per_mile

    return {
        "fuel": float(fuel_per_mile),
        "purchase": float(purchase_per_mile),
        "maintenance": float(maintenance_per_mile),
        "total": float(total),
        "lifetime_miles": float(lifetime_miles),
        "purchase_total": float(v["purchase_cost_usd"]),
    }

def lcod_components_per_mile_base_premium(v, g, residual_cost):
    fe = v["fuel_economy_mi_per_unit"]
    if fe <= 0:
        raise ValueError("Fuel economy must be > 0.")

    lifetime_miles = g["lifetime_years"] * v["planned_miles_per_year"]
    if lifetime_miles <= 0:
        raise ValueError("Lifetime miles must be > 0.")

    fuel_per_mile = (1.0 / fe) * v["fuel_price_usd_per_unit"]
    purchase_total = v["base_price_usd"] + v["premium_price_usd"]
    purchase_per_mile = (purchase_total - residual_cost) / lifetime_miles
    maintenance_per_mile = v["maintenance_usd_per_mile"]
    total = fuel_per_mile + purchase_per_mile + maintenance_per_mile

    return {
        "fuel": float(fuel_per_mile),
        "purchase": float(purchase_per_mile),
        "maintenance": float(maintenance_per_mile),
        "total": float(total),
        "lifetime_miles": float(lifetime_miles),
        "purchase_total": float(purchase_total),
    }


# ============================================================
# ARGONNE-STYLE ECONOMIC DEFAULTS FOR FULL TCO MODEL
# ============================================================
def build_default_economic_inputs() -> Dict[str, Any]:
    """
    Economic add-on defaults for the Full TCO Model.

    Existing LCOD app inputs are NOT replaced. Vehicle cost, fuel price,
    fuel economy, annual miles, maintenance, lifetime, revenue-weight, and
    residual inputs remain from the old app structure.

    Defaults below represent the added Full TCO components using the
    Argonne/Excel-style assumptions discussed for low-advancement cases.
    Application mapping used here:
        Refuse        -> Class 8 Refuse
        Transit Bus   -> Class 8 Refuse assumptions, per user instruction
        Drayage       -> Class 8 Vocational assumptions
        Long Haul     -> Class 8 Sleeper Cab assumptions
    """

    common_finance = {
        "discount_rate": (0.03, 0.03),
        "loan_apr": (0.04, 0.04),
        "down_payment_fraction": (0.12, 0.12),
        "loan_term_years": (5.25, 5.25),
        # Federal corporate income-tax shield settings.
        # A 21% default is used for C corporations; utilization lets users
        # model fleets that cannot use the full tax benefit immediately.
        "corporate_tax_rate": (0.21, 0.21),
        "tax_benefit_utilization_fraction": (1.0, 1.0),
    }

    def veh_payload(diesel=0.0, fcev=0.0, bev=0.0, cng=0.0):
        return {
            "diesel": {
                "payload_penalty_usd_per_mile": (diesel, diesel),
                "fueling_or_charging_labor_usd_per_mile": (0.0, 0.0),
                "annual_afv_registration_usd_per_year": (0.0, 0.0),
            },
            "fcev": {
                "payload_penalty_usd_per_mile": (fcev, fcev),
                "fueling_or_charging_labor_usd_per_mile": (0.0, 0.0),
                "annual_afv_registration_usd_per_year": (0.0, 0.0),
            },
            "bev": {
                "payload_penalty_usd_per_mile": (bev, bev),
                "fueling_or_charging_labor_usd_per_mile": (0.0, 0.0),
                "annual_afv_registration_usd_per_year": (0.0, 0.0),
            },
            "cng": {
                "payload_penalty_usd_per_mile": (cng, cng),
                "fueling_or_charging_labor_usd_per_mile": (0.0, 0.0),
                "annual_afv_registration_usd_per_year": (0.0, 0.0),
            },
        }

    econ = {
        "refuse": {
            "label": "Class 8 Refuse economic assumptions",
            "APP_R": {
                **common_finance,
                "insurance_fixed_usd_per_year": (7500.0, 7500.0),
                "insurance_liability_usd_per_mile": (0.0, 0.0),
                "insurance_physical_damage_usd_per_1000_month": (0.0, 0.0),
                "sales_tax_fraction": (0.0, 0.0),
                "federal_excise_tax_fraction": (0.12, 0.12),
                "initial_registration_usd": (0.0, 0.0),
                "documentation_fee_usd": (0.0, 0.0),
                "annual_registration_usd_per_year": (880.0, 880.0),
                "registration_weight_rate_usd_per_lb_year": (0.0, 0.0),
                "empty_weight_lb": (0.0, 0.0),
                "hvut_weight_rating_lb": (66000.0, 66000.0),
                "hvut_exempt_flag": (0.0, 0.0),
                "permits_licenses_tolls_usd_per_mile": (0.0, 0.0),
                "other_annual_fees_usd_per_year": (0.0, 0.0),
                "driver_labor_usd_per_mile": (2.0231, 2.0231),
            },
            "VEH_R": veh_payload(diesel=0.0, fcev=0.0, bev=0.0517, cng=0.0),
        },

        # Per user instruction, bus uses refuse economic assumptions.
        "bus": {
            "label": "Bus using Class 8 Refuse economic assumptions",
            "APP_R": {
                **common_finance,
                "insurance_fixed_usd_per_year": (7500.0, 7500.0),
                "insurance_liability_usd_per_mile": (0.0, 0.0),
                "insurance_physical_damage_usd_per_1000_month": (0.0, 0.0),
                "sales_tax_fraction": (0.0, 0.0),
                "federal_excise_tax_fraction": (0.12, 0.12),
                "initial_registration_usd": (0.0, 0.0),
                "documentation_fee_usd": (0.0, 0.0),
                "annual_registration_usd_per_year": (880.0, 880.0),
                "registration_weight_rate_usd_per_lb_year": (0.0, 0.0),
                "empty_weight_lb": (0.0, 0.0),
                "hvut_weight_rating_lb": (66000.0, 66000.0),
                "hvut_exempt_flag": (0.0, 0.0),
                "permits_licenses_tolls_usd_per_mile": (0.0, 0.0),
                "other_annual_fees_usd_per_year": (0.0, 0.0),
                "driver_labor_usd_per_mile": (2.0231, 2.0231),
            },
            "VEH_R": veh_payload(diesel=0.0, fcev=0.0, bev=0.0517, cng=0.0),
        },

        # Per user instruction, drayage/short-haul uses Class 8 vocational assumptions.
        "drayage": {
            "label": "Class 8 Vocational economic assumptions",
            "APP_R": {
                **common_finance,
                "insurance_fixed_usd_per_year": (5000.0, 5000.0),
                "insurance_liability_usd_per_mile": (0.0, 0.0),
                "insurance_physical_damage_usd_per_1000_month": (0.0, 0.0),
                "sales_tax_fraction": (0.0, 0.0),
                "federal_excise_tax_fraction": (0.12, 0.12),
                "initial_registration_usd": (0.0, 0.0),
                "documentation_fee_usd": (0.0, 0.0),
                "annual_registration_usd_per_year": (880.0, 880.0),
                "registration_weight_rate_usd_per_lb_year": (0.0, 0.0),
                "empty_weight_lb": (0.0, 0.0),
                "hvut_weight_rating_lb": (66000.0, 66000.0),
                "hvut_exempt_flag": (0.0, 0.0),
                "permits_licenses_tolls_usd_per_mile": (0.0, 0.0),
                "other_annual_fees_usd_per_year": (0.0, 0.0),
                "driver_labor_usd_per_mile": (2.2951, 2.2951),
            },
            "VEH_R": veh_payload(diesel=0.0, fcev=0.0244, bev=0.0503, cng=0.0),
        },

        "longhaul": {
            "label": "Class 8 Sleeper Cab economic assumptions",
            "APP_R": {
                **common_finance,
                "insurance_fixed_usd_per_year": (0.0, 0.0),
                "insurance_liability_usd_per_mile": (0.065, 0.065),
                "insurance_physical_damage_usd_per_1000_month": (2.0, 3.0),
                "sales_tax_fraction": (0.0, 0.0),
                "federal_excise_tax_fraction": (0.12, 0.12),
                "initial_registration_usd": (0.0, 0.0),
                "documentation_fee_usd": (0.0, 0.0),
                "annual_registration_usd_per_year": (1425.0, 1425.0),
                "registration_weight_rate_usd_per_lb_year": (0.0, 0.0),
                "empty_weight_lb": (0.0, 0.0),
                "hvut_weight_rating_lb": (80000.0, 80000.0),
                "hvut_exempt_flag": (0.0, 0.0),
                "permits_licenses_tolls_usd_per_mile": (0.05, 0.05),
                "other_annual_fees_usd_per_year": (0.0, 0.0),
                "driver_labor_usd_per_mile": (0.7900, 0.7900),
            },
            "VEH_R": veh_payload(diesel=0.0, fcev=0.0336, bev=0.2617, cng=0.0),
        },
    }

    return econ


# ============================================================
# FULL TCO CALCULATION FUNCTIONS
# ============================================================
def pv_factor(time_years: float, discount_rate: float) -> float:
    return 1.0 / ((1.0 + discount_rate) ** float(time_years))


def discounted_annual_sum(annual_value: float, lifetime_years: float, discount_rate: float) -> float:
    """PV of a constant annual cost with a fractional final year."""
    life = float(lifetime_years)
    full_years = int(np.floor(life))
    frac = life - full_years

    total = 0.0
    for t in range(1, full_years + 1):
        total += annual_value * pv_factor(t, discount_rate)

    if frac > 1e-12:
        total += frac * annual_value * pv_factor(full_years + 1, discount_rate)

    return float(total)


def discounted_miles_sum(annual_miles: float, lifetime_years: float, discount_rate: float) -> float:
    return discounted_annual_sum(annual_miles, lifetime_years, discount_rate)


def _schedule_length_for_life(lifetime_years: float) -> int:
    """Number of calendar-year values needed, including a possible fractional final year."""
    life = float(lifetime_years)
    if life <= 0:
        return 0
    return int(np.ceil(life - 1e-12))


def _extend_year_schedule(raw_values: list[float], n_years: int) -> list[float]:
    """Return n_years values; repeat the final value if the requested life exceeds the table."""
    if n_years <= 0:
        return []
    if not raw_values:
        return [0.0] * n_years
    values = [float(v) for v in raw_values]
    if n_years <= len(values):
        return values[:n_years]
    return values + [values[-1]] * (n_years - len(values))


def undiscounted_schedule_sum(annual_values: list[float], lifetime_years: float) -> float:
    """Undiscounted sum of a year-by-year schedule with a fractional final year."""
    life = float(lifetime_years)
    full_years = int(np.floor(life))
    frac = life - full_years

    total = 0.0
    for idx in range(full_years):
        if idx < len(annual_values):
            total += float(annual_values[idx])

    if frac > 1e-12 and full_years < len(annual_values):
        total += frac * float(annual_values[full_years])

    return float(total)


def discounted_schedule_sum(annual_values: list[float], lifetime_years: float, discount_rate: float) -> float:
    """PV of a year-by-year annual value schedule with a fractional final year."""
    life = float(lifetime_years)
    full_years = int(np.floor(life))
    frac = life - full_years

    total = 0.0
    for t in range(1, full_years + 1):
        idx = t - 1
        if idx < len(annual_values):
            total += float(annual_values[idx]) * pv_factor(t, discount_rate)

    if frac > 1e-12 and full_years < len(annual_values):
        t = full_years + 1
        total += frac * float(annual_values[full_years]) * pv_factor(t, discount_rate)

    return float(total)


def annual_miles_schedule(
    app_key: str,
    annual_miles: float,
    lifetime_years: float,
    mileage_mode: str,
) -> list[float]:
    """Create the annual-mile schedule used by Full TCO.

    Constant mode repeats the user-entered annual mileage.
    Argonne/VIUS mode uses the approximate Figure 2.7 VMT curve as a shape and
    scales it so the user's selected average annual mileage is preserved over
    the sampled lifetime.
    """
    n_years = _schedule_length_for_life(lifetime_years)
    if n_years <= 0:
        return []

    user_annual = float(annual_miles)
    if mileage_mode != MILEAGE_MODE_VIUS or app_key not in ARGONNE_VIUS_MHDV_VMT:
        return [user_annual] * n_years

    raw_schedule = _extend_year_schedule(ARGONNE_VIUS_MHDV_VMT[app_key], n_years)
    raw_lifetime_miles = undiscounted_schedule_sum(raw_schedule, lifetime_years)
    target_lifetime_miles = user_annual * float(lifetime_years)

    if raw_lifetime_miles <= 0:
        return [user_annual] * n_years

    scale = target_lifetime_miles / raw_lifetime_miles
    return [float(v) * scale for v in raw_schedule]


def mileage_mode_detail(app_key: str, mileage_mode: str) -> str:
    if mileage_mode == MILEAGE_MODE_VIUS:
        return f"{MILEAGE_MODE_VIUS} ({ARGONNE_VIUS_LABELS.get(app_key, 'mapped MHDV schedule')})"
    return MILEAGE_MODE_CONSTANT


def hvut_annual_usd(weight_rating_lb: float, exempt_flag: float) -> float:
    if exempt_flag >= 0.5:
        return 0.0

    w = float(weight_rating_lb)
    if w < 55000.0:
        return 0.0
    if w <= 75500.0:
        return 100.0 + 22.0 * ((w - 55000.0) / 1000.0)
    return 550.0


def pv_financing_interest(
    purchase_cost: float,
    down_payment_fraction: float,
    loan_apr: float,
    loan_term_years: float,
    discount_rate: float,
) -> float:
    """
    Present value of loan-interest payments only.
    The principal is not counted here because vehicle purchase cost is counted
    in the vehicle component.
    """
    loan_amount = purchase_cost * (1.0 - down_payment_fraction)
    if loan_amount <= 0:
        return 0.0

    n_months = int(round(loan_term_years * 12.0))
    if n_months <= 0:
        return 0.0

    monthly_rate = loan_apr / 12.0
    monthly_discount = discount_rate / 12.0

    if abs(monthly_rate) < 1e-12:
        return 0.0

    monthly_payment = loan_amount * (
        monthly_rate * (1.0 + monthly_rate) ** n_months
    ) / ((1.0 + monthly_rate) ** n_months - 1.0)

    balance = loan_amount
    pv_interest = 0.0

    for m in range(1, n_months + 1):
        interest = balance * monthly_rate
        principal = monthly_payment - interest
        balance = max(0.0, balance - principal)
        pv_interest += interest / ((1.0 + monthly_discount) ** m)

    return float(pv_interest)


def estimate_vehicle_value_by_year(
    purchase_cost: float,
    residual_cost: float,
    year: float,
    lifetime_years: float,
) -> float:
    """Linear approximation used only for physical-damage insurance."""
    life = max(1e-9, float(lifetime_years))
    frac = min(max(float(year) / life, 0.0), 1.0)
    return float(purchase_cost - (purchase_cost - residual_cost) * frac)


def pv_physical_damage_insurance(
    purchase_cost: float,
    residual_cost: float,
    lifetime_years: float,
    discount_rate: float,
    rate_usd_per_1000_month: float,
) -> float:
    if rate_usd_per_1000_month <= 0:
        return 0.0

    life = float(lifetime_years)
    full_years = int(np.floor(life))
    frac = life - full_years

    total = 0.0
    for t in range(1, full_years + 1):
        vehicle_value_t = estimate_vehicle_value_by_year(
            purchase_cost=purchase_cost,
            residual_cost=residual_cost,
            year=t,
            lifetime_years=lifetime_years,
        )
        annual_cost = vehicle_value_t / 1000.0 * rate_usd_per_1000_month * 12.0
        total += annual_cost * pv_factor(t, discount_rate)

    if frac > 1e-12:
        t = full_years + 1
        vehicle_value_t = estimate_vehicle_value_by_year(
            purchase_cost=purchase_cost,
            residual_cost=residual_cost,
            year=t,
            lifetime_years=lifetime_years,
        )
        annual_cost = vehicle_value_t / 1000.0 * rate_usd_per_1000_month * 12.0
        total += frac * annual_cost * pv_factor(t, discount_rate)

    return float(total)



def clamp_fraction(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(float(value), lo), hi))


def macrs_5yr_half_year_rates() -> list[float]:
    """
    Standard 5-year MACRS half-year-convention percentages.
    The sixth entry appears because the half-year convention spreads recovery
    over six tax years.
    """
    return [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]


def pv_depreciation_tax_shield(
    depreciable_basis: float,
    lifetime_years: float,
    discount_rate: float,
    corporate_tax_rate: float,
    utilization_fraction: float,
) -> float:
    """PV of tax benefit from 5-year MACRS depreciation.

    Purchase cost is not deducted immediately here. Instead, the tax benefit
    comes from depreciation deductions. If the modeled vehicle life ends before
    all MACRS years are used, only deduction years inside the operating life are
    counted.
    """
    rate = max(0.0, float(corporate_tax_rate))
    util = clamp_fraction(utilization_fraction)
    life = float(lifetime_years)

    pv_tax = 0.0
    for year, macrs_rate in enumerate(macrs_5yr_half_year_rates(), start=1):
        if year - 1 >= life:
            break
        depreciation_deduction = float(depreciable_basis) * macrs_rate
        pv_tax += depreciation_deduction * rate * util * pv_factor(year, discount_rate)

    return float(pv_tax)


def federal_tax_benefit_pv(
    purchase_total: float,
    financing_interest_pv: float,
    deductible_operating_pv: float,
    lifetime_years: float,
    discount_rate: float,
    corporate_tax_rate: float,
    utilization_fraction: float,
) -> Dict[str, float]:
    """PV of federal corporate income-tax shields.

    This treats recurring operating costs and loan interest as deductible and
    treats vehicle purchase cost through depreciation, not as an immediate
    full deduction.
    """
    rate = max(0.0, float(corporate_tax_rate))
    util = clamp_fraction(utilization_fraction)

    depreciation_tax_shield = pv_depreciation_tax_shield(
        depreciable_basis=purchase_total,
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
        corporate_tax_rate=rate,
        utilization_fraction=util,
    )
    financing_tax_shield = float(financing_interest_pv) * rate * util
    operating_tax_shield = float(deductible_operating_pv) * rate * util

    total = depreciation_tax_shield + financing_tax_shield + operating_tax_shield

    return {
        "depreciation_tax_shield": float(depreciation_tax_shield),
        "financing_tax_shield": float(financing_tax_shield),
        "operating_tax_shield": float(operating_tax_shield),
        "total_tax_benefit": float(total),
    }

def full_tco_components_per_mile(
    v,
    g,
    e,
    residual_cost: float,
    purchase_total: float,
    app_key: str,
    mileage_mode: str,
) -> Dict[str, Any]:
    """
    Full TCO LCOD = PV(all cost components) / PV(miles).
    Vehicle/fuel/maintenance inputs come from the original LCOD app.
    Added economic inputs come from the Economic and Other Inputs sidebar.
    """
    discount_rate = float(e["discount_rate"])
    lifetime_years = float(g["lifetime_years"])
    annual_miles = float(v["planned_miles_per_year"])

    if annual_miles <= 0:
        raise ValueError("Annual miles must be > 0.")
    if lifetime_years <= 0:
        raise ValueError("Lifetime years must be > 0.")
    if v["fuel_economy_mi_per_unit"] <= 0:
        raise ValueError("Fuel economy must be > 0.")

    miles_by_year = annual_miles_schedule(
        app_key=app_key,
        annual_miles=annual_miles,
        lifetime_years=lifetime_years,
        mileage_mode=mileage_mode,
    )

    discounted_miles = discounted_schedule_sum(miles_by_year, lifetime_years, discount_rate)
    undiscounted_miles = undiscounted_schedule_sum(miles_by_year, lifetime_years)
    if discounted_miles <= 0:
        raise ValueError("Discounted miles must be > 0.")

    # Vehicle component: upfront purchase minus PV residual value.
    vehicle_pv = purchase_total - residual_cost * pv_factor(lifetime_years, discount_rate)

    # Financing component: PV of interest only.
    financing_pv = pv_financing_interest(
        purchase_cost=purchase_total,
        down_payment_fraction=e["down_payment_fraction"],
        loan_apr=e["loan_apr"],
        loan_term_years=e["loan_term_years"],
        discount_rate=discount_rate,
    )

    # Fuel and maintenance from original LCOD app inputs.
    fuel_per_mile = v["fuel_price_usd_per_unit"] / v["fuel_economy_mi_per_unit"]
    fuel_pv = discounted_schedule_sum(
        [fuel_per_mile * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )

    maintenance_per_mile = v["maintenance_usd_per_mile"]
    maintenance_pv = discounted_schedule_sum(
        [maintenance_per_mile * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )

    # Insurance: fixed annual + liability mileage-based + physical-damage value-based.
    fixed_insurance_pv = discounted_annual_sum(
        annual_value=e["insurance_fixed_usd_per_year"],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )
    liability_insurance_pv = discounted_schedule_sum(
        [e["insurance_liability_usd_per_mile"] * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )
    physical_damage_pv = pv_physical_damage_insurance(
        purchase_cost=purchase_total,
        residual_cost=residual_cost,
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
        rate_usd_per_1000_month=e["insurance_physical_damage_usd_per_1000_month"],
    )
    insurance_pv = fixed_insurance_pv + liability_insurance_pv + physical_damage_pv

    # Taxes and fees.
    upfront_tax_fees = (
        purchase_total * e["sales_tax_fraction"]
        + purchase_total * e["federal_excise_tax_fraction"]
        + e["initial_registration_usd"]
        + e["documentation_fee_usd"]
    )

    hvut = hvut_annual_usd(e["hvut_weight_rating_lb"], e["hvut_exempt_flag"])
    annual_registration = (
        e["annual_registration_usd_per_year"]
        + e["annual_afv_registration_usd_per_year"]
        + e["registration_weight_rate_usd_per_lb_year"] * e["empty_weight_lb"]
    )
    fixed_annual_tax_fees = hvut + annual_registration + e["other_annual_fees_usd_per_year"]
    fixed_annual_tax_fees_pv = discounted_annual_sum(fixed_annual_tax_fees, lifetime_years, discount_rate)
    permits_tolls_pv = discounted_schedule_sum(
        [e["permits_licenses_tolls_usd_per_mile"] * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )
    annual_tax_fees_pv = fixed_annual_tax_fees_pv + permits_tolls_pv
    tax_fees_pv = upfront_tax_fees + annual_tax_fees_pv

    # Payload penalty and labor.
    payload_pv = discounted_schedule_sum(
        [e["payload_penalty_usd_per_mile"] * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )

    labor_per_mile = e["driver_labor_usd_per_mile"] + e["fueling_or_charging_labor_usd_per_mile"]
    labor_pv = discounted_schedule_sum(
        [labor_per_mile * m for m in miles_by_year],
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
    )

    pv_components = {
        "vehicle": float(vehicle_pv),
        "financing": float(financing_pv),
        "fuel": float(fuel_pv),
        "maintenance": float(maintenance_pv),
        "insurance": float(insurance_pv),
        "tax_fees": float(tax_fees_pv),
        "payload": float(payload_pv),
        "labor": float(labor_pv),
    }

    mile_components = {k: pv_components[k] / discounted_miles for k in FULL_TCO_COMPONENTS}
    total_mile = sum(mile_components.values())

    # Federal corporate income-tax benefit.
    # Operating deductions include recurring business costs and annual taxes/fees.
    # Upfront vehicle purchase cost is handled through MACRS depreciation instead
    # of being deducted immediately.
    deductible_operating_pv = (
        fuel_pv
        + maintenance_pv
        + insurance_pv
        + annual_tax_fees_pv
        + payload_pv
        + labor_pv
    )
    tax_pv_components = federal_tax_benefit_pv(
        purchase_total=purchase_total,
        financing_interest_pv=financing_pv,
        deductible_operating_pv=deductible_operating_pv,
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
        corporate_tax_rate=e.get("corporate_tax_rate", 0.0),
        utilization_fraction=e.get("tax_benefit_utilization_fraction", 0.0),
    )

    federal_tax_benefit = tax_pv_components["total_tax_benefit"]
    after_tax_total_mile = total_mile - federal_tax_benefit / discounted_miles

    return {
        "pv_components": pv_components,
        "mile_components": mile_components,
        "total_mile": float(total_mile),
        "tax_pv_components": tax_pv_components,
        "federal_tax_benefit_pv": float(federal_tax_benefit),
        "federal_tax_benefit_mile": float(federal_tax_benefit / discounted_miles),
        "after_tax_total_mile": float(after_tax_total_mile),
        "discounted_miles": float(discounted_miles),
        "undiscounted_miles": float(undiscounted_miles),
    }


def sample_economic_inputs(econ_cfg: Dict[str, Any], vt: str, rng: random.Random) -> Dict[str, float]:
    e = {
        k: _sample_uniform(rng, r)
        for k, r in econ_cfg["APP_R"].items()
        if _is_range(r)
    }

    if vt in econ_cfg["VEH_R"]:
        for k, r in econ_cfg["VEH_R"][vt].items():
            if _is_range(r):
                e[k] = _sample_uniform(rng, r)

    return e


# ============================================================
# RUN APPLICATION MODEL
# Original LCOD and breakeven inputs are preserved.
# Full TCO outputs are added separately.
# ============================================================
def run_application_model(
    app_name: str,
    app_cfg: Dict[str, Any],
    econ_cfg: Dict[str, Any],
    econ_mode: str,
    mileage_mode: str,
    n_samples: int,
    random_seed: int,
) -> Dict[str, Any]:

    GLOBAL_R = app_cfg["GLOBAL_R"]
    VEH_R = app_cfg["VEH_R"]
    RESIDUAL_R = app_cfg["RESIDUAL_R"]
    price_mode = app_cfg["price_mode"]

    veh_list = [vt for vt in VEHICLE_ORDER if vehicle_is_complete(app_cfg, vt)]

    rng = random.Random(int(random_seed))
    N = int(n_samples)

    # Original LCOD arrays, unchanged.
    lcod_mile = {vt: [] for vt in veh_list}
    lcod_tm = {vt: [] for vt in veh_list}

    # New Full TCO arrays.
    full_tco_mile = {vt: [] for vt in veh_list}
    full_tco_tm = {vt: [] for vt in veh_list}
    full_tco_total_pv = {vt: [] for vt in veh_list}
    full_tco_component_mile = {
        vt: {comp: [] for comp in FULL_TCO_COMPONENTS}
        for vt in veh_list
    }
    full_tco_component_tm = {
        vt: {comp: [] for comp in FULL_TCO_COMPONENTS}
        for vt in veh_list
    }
    full_tco_component_total_pv = {
        vt: {comp: [] for comp in FULL_TCO_COMPONENTS}
        for vt in veh_list
    }
    after_tax_full_tco_mile = {vt: [] for vt in veh_list}
    after_tax_full_tco_tm = {vt: [] for vt in veh_list}
    after_tax_full_tco_total_pv = {vt: [] for vt in veh_list}
    federal_tax_benefit_mile = {vt: [] for vt in veh_list}
    federal_tax_benefit_tm = {vt: [] for vt in veh_list}
    federal_tax_benefit_total_pv = {vt: [] for vt in veh_list}
    federal_tax_benefit_component_mile = {
        vt: {
            "depreciation_tax_shield": [],
            "financing_tax_shield": [],
            "operating_tax_shield": [],
        }
        for vt in veh_list
    }
    federal_tax_benefit_component_total_pv = {
        vt: {
            "depreciation_tax_shield": [],
            "financing_tax_shield": [],
            "operating_tax_shield": [],
        }
        for vt in veh_list
    }
    discounted_miles = {vt: [] for vt in veh_list}
    undiscounted_miles = {vt: [] for vt in veh_list}

    breakeven_inputs = {
        vt: {
            "fuel_per_mile": [],
            "maintenance_per_mile": [],
            "lifetime_miles": [],
            "residual_cost": [],
            "purchase_total": [],
        }
        for vt in veh_list
    }

    for _ in range(N):
        g = {k: _sample_uniform(rng, r) for k, r in GLOBAL_R.items()}

        batt_mass = bev_battery_mass_kg(
            g["bev_battery_energy_kwh"],
            g["bev_energy_density_kwh_per_kg"]
        )

        rev_bev = bev_revenue_weight_ton(
            g["revenue_weight_ton_diesel_fcev"],
            batt_mass
        )

        for vt in veh_list:
            v = {
                k: _sample_uniform(rng, rv)
                for k, rv in VEH_R[vt].items()
                if rv is not None and _is_range(rv)
            }

            residual_tbl = {vt: {}}
            for cname, fields in RESIDUAL_R[vt].items():
                residual_tbl[vt][cname] = {
                    rk: _sample_uniform(rng, fields[rk])
                    for rk in REQUIRED_RES_KEYS
                }

            residual_cost = residual_cost_usd_point(residual_tbl, vt, "current")

            if price_mode == "single_purchase":
                comp = lcod_components_per_mile_single_purchase(v, g, residual_cost)
            else:
                comp = lcod_components_per_mile_base_premium(v, g, residual_cost)

            rev_wt = rev_bev if vt == "bev" else g["revenue_weight_ton_diesel_fcev"]

            # Original LCOD calculation, unchanged.
            lmi = comp["total"]
            ltm = lcod_usd_per_ton_mile_point(lmi, rev_wt)
            lcod_mile[vt].append(lmi)
            lcod_tm[vt].append(ltm)

            # Breakeven inputs remain based on original LCOD logic.
            breakeven_inputs[vt]["fuel_per_mile"].append(comp["fuel"])
            breakeven_inputs[vt]["maintenance_per_mile"].append(comp["maintenance"])
            breakeven_inputs[vt]["lifetime_miles"].append(comp["lifetime_miles"])
            breakeven_inputs[vt]["residual_cost"].append(residual_cost)
            breakeven_inputs[vt]["purchase_total"].append(comp["purchase_total"])

            # Full TCO calculation.
            e = sample_economic_inputs(econ_cfg, vt, rng)
            full = full_tco_components_per_mile(
                v=v,
                g=g,
                e=e,
                residual_cost=residual_cost,
                purchase_total=comp["purchase_total"],
                app_key=app_name,
                mileage_mode=mileage_mode,
            )

            full_mile = full["total_mile"]
            full_tm = lcod_usd_per_ton_mile_point(full_mile, rev_wt)
            full_total_pv = sum(full["pv_components"].values())

            after_tax_mile = full["after_tax_total_mile"]
            after_tax_tm = lcod_usd_per_ton_mile_point(after_tax_mile, rev_wt)
            tax_benefit_mile = full["federal_tax_benefit_mile"]
            tax_benefit_tm = lcod_usd_per_ton_mile_point(tax_benefit_mile, rev_wt)
            tax_benefit_total_pv = full["federal_tax_benefit_pv"]
            after_tax_total_pv = full_total_pv - tax_benefit_total_pv

            full_tco_mile[vt].append(full_mile)
            full_tco_tm[vt].append(full_tm)
            full_tco_total_pv[vt].append(full_total_pv)
            after_tax_full_tco_mile[vt].append(after_tax_mile)
            after_tax_full_tco_tm[vt].append(after_tax_tm)
            after_tax_full_tco_total_pv[vt].append(after_tax_total_pv)
            federal_tax_benefit_mile[vt].append(tax_benefit_mile)
            federal_tax_benefit_tm[vt].append(tax_benefit_tm)
            federal_tax_benefit_total_pv[vt].append(tax_benefit_total_pv)
            discounted_miles[vt].append(full["discounted_miles"])
            undiscounted_miles[vt].append(full["undiscounted_miles"])

            for cname in FULL_TCO_COMPONENTS:
                c_mile = full["mile_components"][cname]
                c_total_pv = full["pv_components"][cname]
                full_tco_component_mile[vt][cname].append(c_mile)
                full_tco_component_tm[vt][cname].append(c_mile / rev_wt)
                full_tco_component_total_pv[vt][cname].append(c_total_pv)

            for cname in federal_tax_benefit_component_mile[vt]:
                c_total_pv = full["tax_pv_components"][cname]
                c_mile = c_total_pv / full["discounted_miles"]
                federal_tax_benefit_component_mile[vt][cname].append(c_mile)
                federal_tax_benefit_component_total_pv[vt][cname].append(c_total_pv)

    return {
        "application": app_name,
        "label": app_cfg["label"],
        "economic_label": econ_cfg.get("label", "Economic inputs"),
        "economic_mode": econ_mode,
        "mileage_mode": mileage_mode,
        "mileage_mode_detail": mileage_mode_detail(app_name, mileage_mode),
        "VEH_LIST": veh_list,
        "lcod_mile": lcod_mile,
        "lcod_tm": lcod_tm,
        "breakeven_inputs": breakeven_inputs,
        "full_tco_mile": full_tco_mile,
        "full_tco_tm": full_tco_tm,
        "full_tco_total_pv": full_tco_total_pv,
        "full_tco_component_mile": full_tco_component_mile,
        "full_tco_component_tm": full_tco_component_tm,
        "full_tco_component_total_pv": full_tco_component_total_pv,
        "after_tax_full_tco_mile": after_tax_full_tco_mile,
        "after_tax_full_tco_tm": after_tax_full_tco_tm,
        "after_tax_full_tco_total_pv": after_tax_full_tco_total_pv,
        "federal_tax_benefit_mile": federal_tax_benefit_mile,
        "federal_tax_benefit_tm": federal_tax_benefit_tm,
        "federal_tax_benefit_total_pv": federal_tax_benefit_total_pv,
        "federal_tax_benefit_component_mile": federal_tax_benefit_component_mile,
        "federal_tax_benefit_component_total_pv": federal_tax_benefit_component_total_pv,
        "discounted_miles": discounted_miles,
        "undiscounted_miles": undiscounted_miles,
    }


# ============================================================
# STREAMLIT CONTROLS
# ============================================================
DEFAULT_APPLICATIONS = add_residual_tables(build_base_applications())
APPLICATIONS = copy.deepcopy(
    st.session_state.get("APPLICATIONS", DEFAULT_APPLICATIONS)
)

DEFAULT_ECON_R = build_default_economic_inputs()
DEFAULT_ECON_MODE = {app_key: ECON_MODE_DEFAULT for app_key in APP_ORDER}


def merge_missing_defaults(user_cfg: Dict[str, Any], default_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve existing/custom values while adding new v7 default keys."""
    merged = copy.deepcopy(default_cfg)

    def _merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    if isinstance(user_cfg, dict):
        _merge(merged, user_cfg)
    return merged


# Draft economic inputs are the values currently shown/edited in the sidebar.
# They are NOT used in the displayed results until the user presses Run Full TCO Model.
PENDING_ECON_R = merge_missing_defaults(
    st.session_state.get("PENDING_ECON_R", DEFAULT_ECON_R),
    DEFAULT_ECON_R,
)
PENDING_ECON_MODE = merge_missing_defaults(
    st.session_state.get("PENDING_ECON_MODE", DEFAULT_ECON_MODE),
    DEFAULT_ECON_MODE,
)

# Local names used by the economic sidebar below. These are pending values only.
ECON_R = copy.deepcopy(PENDING_ECON_R)
ECON_MODE = copy.deepcopy(PENDING_ECON_MODE)

# ============================================================
# SIDEBAR APP SELECTOR FOR OLD LCOD INPUT EDITING
# ============================================================
st.sidebar.header("Edit Existing LCOD Input Ranges")

edit_app_key = st.sidebar.selectbox(
    "Application to edit",
    options=APP_ORDER,
    index=APP_ORDER.index(st.session_state.get("edit_app_key", "refuse")),
    format_func=lambda x: APPLICATIONS[x]["label"],
    key="edit_app_selector",
)

st.session_state["edit_app_key"] = edit_app_key
edit_app = APPLICATIONS[edit_app_key]

# ============================================================
# SIDEBAR FORM
# ============================================================
with st.sidebar.form("model_input_form"):

    st.header("Run Settings")

    N_SAMPLES = st.slider(
        "Monte Carlo samples",
        min_value=1000,
        max_value=50000,
        value=int(st.session_state.get("N_SAMPLES", DEFAULT_N_SAMPLES)),
        step=1000,
    )

    RANDOM_SEED = st.number_input(
        "Random seed",
        value=int(st.session_state.get("RANDOM_SEED", DEFAULT_RANDOM_SEED)),
        step=1,
    )

    previous_mileage_mode = st.session_state.get("MILEAGE_MODE", MILEAGE_MODE_CONSTANT)
    if previous_mileage_mode not in MILEAGE_MODE_OPTIONS:
        previous_mileage_mode = MILEAGE_MODE_CONSTANT

    MILEAGE_MODE = st.radio(
        "Full TCO mileage method",
        options=MILEAGE_MODE_OPTIONS,
        index=MILEAGE_MODE_OPTIONS.index(previous_mileage_mode),
        help=(
            "Constant mode repeats the user-entered annual mileage. "
            "Argonne/VIUS mode uses approximate year-by-year mileage shapes from Argonne Figure 2.7, "
            "scaled to preserve the user-entered average annual mileage. "
            "The original LCOD breakeven calculation is kept unchanged."
        ),
    )

    st.markdown("---")
    st.markdown(f"### Editing existing LCOD inputs: {edit_app['label']}")

    with st.expander(f"{edit_app['label']} — Global Inputs", expanded=False):
        for k, r in edit_app["GLOBAL_R"].items():
            APPLICATIONS[edit_app_key]["GLOBAL_R"][k] = range_input(
                k,
                r,
                f"{edit_app_key}_global_{k}"
            )

    for vt in VEHICLE_ORDER:
        if not vehicle_is_complete(APPLICATIONS[edit_app_key], vt):
            continue

        with st.expander(f"{edit_app['label']} — {vt.upper()} Existing Inputs", expanded=False):

            st.markdown("**Vehicle inputs from old LCOD app**")
            for k, r in edit_app["VEH_R"][vt].items():
                APPLICATIONS[edit_app_key]["VEH_R"][vt][k] = range_input(
                    k,
                    r,
                    f"{edit_app_key}_{vt}_{k}"
                )

            st.markdown("**Residual inputs from old LCOD app**")
            for cname, fields in edit_app["RESIDUAL_R"][vt].items():
                st.markdown(f"_{cname}_")
                for k, r in fields.items():
                    APPLICATIONS[edit_app_key]["RESIDUAL_R"][vt][cname][k] = range_input(
                        k,
                        r,
                        f"{edit_app_key}_{vt}_{cname}_{k}"
                    )

    st.markdown("---")
    st.markdown("### Run one application case")
    st.caption(
        "Only the clicked application is calculated and displayed. "
        "This keeps the app lighter on Streamlit Cloud."
    )

    run_app_buttons = {}
    for _app_key in APP_ORDER:
        run_app_buttons[_app_key] = st.form_submit_button(
            f"Run {APPLICATIONS[_app_key]['label']} Case",
            width="stretch",
        )

    run_app_key = next(
        (_app_key for _app_key, _pressed in run_app_buttons.items() if _pressed),
        None,
    )
    run_button = run_app_key is not None

# ============================================================
# ECONOMIC INPUTS OUTSIDE FORM
# This makes custom input boxes appear immediately after selecting custom.
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.header("Economic and Other Inputs")
    st.caption(
        "Select default or custom economic inputs separately for each application. "
        "Custom values are application-specific and do not change defaults for other applications. "
        "Changing these widgets only creates pending edits; results update only after pressing a Run Case button."
    )

    for app_key in APP_ORDER:
        app_label = APPLICATIONS[app_key]["label"]
        default_label = DEFAULT_ECON_R[app_key]["label"]

        with st.expander(f"{app_label} — Economic inputs", expanded=False):
            previous_mode = ECON_MODE.get(app_key, ECON_MODE_DEFAULT)
            mode = st.radio(
                "Economic input mode",
                options=ECON_MODE_OPTIONS,
                index=ECON_MODE_OPTIONS.index(previous_mode),
                key=f"{app_key}_economic_mode",
            )
            ECON_MODE[app_key] = mode

            if mode == ECON_MODE_DEFAULT:
                ECON_R[app_key] = copy.deepcopy(DEFAULT_ECON_R[app_key])
                st.info(f"Using default set: {default_label}")
                show_economic_values_table(
                    DEFAULT_ECON_R[app_key],
                    "Default economic values used"
                )
            else:
                # If custom is selected for the first time, start from defaults.
                if app_key not in ECON_R:
                    ECON_R[app_key] = copy.deepcopy(DEFAULT_ECON_R[app_key])

                st.warning("Custom values are active for this application and will be marked as Customized in plots/tables.")

                with st.expander("Show default reference values", expanded=False):
                    show_economic_values_table(
                        DEFAULT_ECON_R[app_key],
                        "Default reference values"
                    )

                st.markdown("**Application-level economic inputs**")

                for k, r in ECON_R[app_key]["APP_R"].items():
                    ECON_R[app_key]["APP_R"][k] = smart_econ_range_input(
                        k,
                        r,
                        f"econ_{app_key}_app_{k}",
                    )

                st.markdown("**Vehicle-specific economic inputs**")
                for vt in VEHICLE_ORDER:
                    if vt not in ECON_R[app_key]["VEH_R"]:
                        continue
                    if vt not in APPLICATIONS[app_key]["VEH_R"]:
                        continue
                    st.markdown(f"_{vt.upper()}_")
                    for k, r in ECON_R[app_key]["VEH_R"][vt].items():
                        ECON_R[app_key]["VEH_R"][vt][k] = smart_econ_range_input(
                            k,
                            r,
                            f"econ_{app_key}_{vt}_{k}",
                        )



# Save economic sidebar state as DRAFT/PENDING values only.
# These edits persist across reruns but are not applied to results until Run is pressed.
st.session_state["PENDING_ECON_R"] = copy.deepcopy(ECON_R)
st.session_state["PENDING_ECON_MODE"] = copy.deepcopy(ECON_MODE)

# ============================================================
# SIDEBAR DISPLAY OPTIONS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Display Options")

active_app_key_from_state = st.session_state.get("ACTIVE_APP_KEY", None)
if active_app_key_from_state in APP_ORDER:
    st.sidebar.info(
        f"Active result: {APPLICATIONS[active_app_key_from_state]['label']}. "
        "Click another case button to calculate and show a different application."
    )
else:
    st.sidebar.info("No active result yet. Click one of the four Run Case buttons.")

# The app now displays one active application at a time to reduce memory/load.
selected_apps = [active_app_key_from_state] if active_app_key_from_state in APP_ORDER else []

selected_vehicles_display = st.sidebar.multiselect(
    "Vehicles to display",
    options=VEHICLE_ORDER,
    default=st.session_state.get("selected_vehicles_display", VEHICLE_ORDER),
    format_func=lambda x: x.upper(),
)

metric_display_list = st.sidebar.multiselect(
    "Full TCO plot metrics to display",
    options=["$/mile", "Total PV TCO ($)", "$/ton-mile"],
    default=st.session_state.get("metric_display_list", ["$/mile"]),
    help=(
        "You can select more than one metric, but plots are generated only for the active application case. "
        "$/ton-mile is shown only for Drayage and Long Haul."
    ),
)
if len(metric_display_list) == 0:
    st.sidebar.warning("Select at least one plot metric to show Full TCO plots.")

if "$/ton-mile" in metric_display_list:
    non_freight_selected = [app for app in selected_apps if app not in FREIGHT_TON_MILE_APPS]
    if non_freight_selected:
        st.sidebar.info(
            "$/ton-mile plots and table columns are shown only for Drayage and Long Haul. "
            "Refuse and Transit Bus will be skipped for this metric."
        )

st.session_state["metric_display_list"] = metric_display_list

tax_plot_display = st.sidebar.radio(
    "Full TCO tax plots",
    options=["Show both", "Pre-tax only", "After-tax only"],
    index=0,
)

# ============================================================
# RUN OR LOAD MODEL RESULTS
# ============================================================
if run_button:

    if run_app_key not in APP_ORDER:
        st.warning("Click one of the application case buttons to run the model.")
        st.stop()

    if len(selected_vehicles_display) == 0:
        st.warning("Select at least one vehicle.")
        st.stop()

    all_results = {}

    try:
        # Calculate only the clicked application. Other applications are not recalculated.
        all_results[run_app_key] = run_application_model(
            app_name=run_app_key,
            app_cfg=APPLICATIONS[run_app_key],
            econ_cfg=ECON_R[run_app_key],
            econ_mode=ECON_MODE[run_app_key],
            mileage_mode=MILEAGE_MODE,
            n_samples=N_SAMPLES,
            random_seed=RANDOM_SEED,
        )

        st.session_state["all_results"] = all_results
        st.session_state["ACTIVE_APP_KEY"] = run_app_key
        st.session_state["APPLICATIONS"] = copy.deepcopy(APPLICATIONS)
        # These are the active economic inputs used for the completed run.
        st.session_state["ACTIVE_ECON_R"] = copy.deepcopy(ECON_R)
        st.session_state["ACTIVE_ECON_MODE"] = copy.deepcopy(ECON_MODE)
        st.session_state["N_SAMPLES"] = int(N_SAMPLES)
        st.session_state["RANDOM_SEED"] = int(RANDOM_SEED)
        st.session_state["MILEAGE_MODE"] = MILEAGE_MODE
        st.session_state["LAST_RUN_MILEAGE_MODE"] = MILEAGE_MODE
        st.session_state["selected_apps"] = [run_app_key]
        st.session_state["selected_vehicles_display"] = selected_vehicles_display
        st.session_state["LAST_RUN_ECON_R"] = copy.deepcopy(ECON_R)
        st.session_state["LAST_RUN_ECON_MODE"] = copy.deepcopy(ECON_MODE)

        selected_apps = [run_app_key]

    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

else:

    if "all_results" not in st.session_state:
        st.info("Adjust sidebar inputs and click one of the four **Run Case** buttons to calculate results.")
        st.stop()

    all_results = st.session_state["all_results"]

    old_result_format = any(
        "after_tax_full_tco_mile" not in res
        or "federal_tax_benefit_mile" not in res
        or "full_tco_total_pv" not in res
        or "after_tax_full_tco_total_pv" not in res
        or "mileage_mode" not in res
        or "undiscounted_miles" not in res
        for res in all_results.values()
    )
    if old_result_format:
        st.info(
            "Previous results were generated by an older app version. "
            "Click one of the four **Run Case** buttons to calculate the new pre-tax, after-tax, total PV TCO, and mileage-mode results."
        )
        st.stop()

    APPLICATIONS = st.session_state["APPLICATIONS"]

    # Keep current sidebar economic values as pending edits.
    # Do not replace them with last-run values here, otherwise the UI would lose custom edits.
    # The displayed tables/plots below come from all_results, which remains from the last completed run.

    active_app_key = st.session_state.get("ACTIVE_APP_KEY", None)
    if active_app_key not in all_results:
        if len(all_results) > 0:
            active_app_key = next(iter(all_results.keys()))
            st.session_state["ACTIVE_APP_KEY"] = active_app_key
        else:
            st.info("Click one of the four **Run Case** buttons to calculate results.")
            st.stop()

    selected_apps = [active_app_key]
    st.session_state["selected_apps"] = selected_apps
    st.session_state["selected_vehicles_display"] = selected_vehicles_display

if len(selected_apps) == 0 or len(selected_vehicles_display) == 0:
    st.warning("Select at least one vehicle and run one application case.")
    st.stop()

active_label = APPLICATIONS[selected_apps[0]]["label"]
st.success(
    f"Full TCO results are loaded for {active_label}. "
    "Display filters and breakeven selectors can be changed without rerunning the Monte Carlo simulation."
)

loaded_mileage_modes = sorted({res.get("mileage_mode", MILEAGE_MODE_CONSTANT) for res in all_results.values()})
st.info("Mileage method used in loaded Full TCO results: " + ", ".join(loaded_mileage_modes))

pending_econ_changes = False
if "LAST_RUN_ECON_R" in st.session_state and "LAST_RUN_ECON_MODE" in st.session_state:
    pending_econ_changes = (
        st.session_state.get("PENDING_ECON_R", DEFAULT_ECON_R) != st.session_state["LAST_RUN_ECON_R"]
        or st.session_state.get("PENDING_ECON_MODE", DEFAULT_ECON_MODE) != st.session_state["LAST_RUN_ECON_MODE"]
    )

if pending_econ_changes:
    st.warning(
        "Economic input values have been edited after the last completed model run. "
        "The tables and plots below are frozen from the previous run. Click the relevant **Run Case** button to apply the pending edits."
    )
else:
    st.info(
        "Economic input edits are stored as pending values and will not be applied until you click a **Run Case** button."
    )

# ============================================================
# FULL TCO SUMMARY TABLES
# ============================================================
st.subheader("Full TCO Model Summary Tables")

all_full_rows = []
all_component_rows = []

for app_key in selected_apps:
    res = all_results[app_key]
    app_mode_label = econ_input_set_label(res["economic_mode"])
    rows = []
    component_rows = []

    for vt in res["VEH_LIST"]:
        if vt not in selected_vehicles_display:
            continue

        mile_s = _summarize_percentiles(np.array(res["full_tco_mile"][vt], dtype=float), PCTILES)
        after_mile_s = _summarize_percentiles(np.array(res["after_tax_full_tco_mile"][vt], dtype=float), PCTILES)
        tax_mile_s = _summarize_percentiles(np.array(res["federal_tax_benefit_mile"][vt], dtype=float), PCTILES)
        total_s = _summarize_percentiles(np.array(res["full_tco_total_pv"][vt], dtype=float), PCTILES)
        after_total_s = _summarize_percentiles(np.array(res["after_tax_full_tco_total_pv"][vt], dtype=float), PCTILES)
        tax_total_s = _summarize_percentiles(np.array(res["federal_tax_benefit_total_pv"][vt], dtype=float), PCTILES)
        discounted_miles_s = _summarize_percentiles(np.array(res["discounted_miles"][vt], dtype=float), PCTILES)
        undiscounted_miles_s = _summarize_percentiles(np.array(res["undiscounted_miles"][vt], dtype=float), PCTILES)

        row = {
            "Application": res["label"],
            "Economic inputs": app_mode_label,
            "Mileage method": res.get("mileage_mode", MILEAGE_MODE_CONSTANT),
            "Vehicle": vt.upper(),
            "P50 undiscounted miles": undiscounted_miles_s["p50"],
            "P50 discounted miles": discounted_miles_s["p50"],
            "P5 Pre-tax Full TCO ($/mile)": mile_s["p5"],
            "P50 Pre-tax Full TCO ($/mile)": mile_s["p50"],
            "P95 Pre-tax Full TCO ($/mile)": mile_s["p95"],
            "P5 Federal tax benefit ($/mile)": tax_mile_s["p5"],
            "P50 Federal tax benefit ($/mile)": tax_mile_s["p50"],
            "P95 Federal tax benefit ($/mile)": tax_mile_s["p95"],
            "P5 After-tax Full TCO ($/mile)": after_mile_s["p5"],
            "P50 After-tax Full TCO ($/mile)": after_mile_s["p50"],
            "P95 After-tax Full TCO ($/mile)": after_mile_s["p95"],
            "P5 Pre-tax Total PV TCO ($)": total_s["p5"],
            "P50 Pre-tax Total PV TCO ($)": total_s["p50"],
            "P95 Pre-tax Total PV TCO ($)": total_s["p95"],
            "P5 Federal tax benefit Total PV ($)": tax_total_s["p5"],
            "P50 Federal tax benefit Total PV ($)": tax_total_s["p50"],
            "P95 Federal tax benefit Total PV ($)": tax_total_s["p95"],
            "P5 After-tax Total PV TCO ($)": after_total_s["p5"],
            "P50 After-tax Total PV TCO ($)": after_total_s["p50"],
            "P95 After-tax Total PV TCO ($)": after_total_s["p95"],
        }

        if app_key in FREIGHT_TON_MILE_APPS:
            tm_s = _summarize_percentiles(np.array(res["full_tco_tm"][vt], dtype=float), PCTILES)
            after_tm_s = _summarize_percentiles(np.array(res["after_tax_full_tco_tm"][vt], dtype=float), PCTILES)
            tax_tm_s = _summarize_percentiles(np.array(res["federal_tax_benefit_tm"][vt], dtype=float), PCTILES)
            row.update({
                "P5 Pre-tax Full TCO ($/ton-mile)": tm_s["p5"],
                "P50 Pre-tax Full TCO ($/ton-mile)": tm_s["p50"],
                "P95 Pre-tax Full TCO ($/ton-mile)": tm_s["p95"],
                "P5 Federal tax benefit ($/ton-mile)": tax_tm_s["p5"],
                "P50 Federal tax benefit ($/ton-mile)": tax_tm_s["p50"],
                "P95 Federal tax benefit ($/ton-mile)": tax_tm_s["p95"],
                "P5 After-tax Full TCO ($/ton-mile)": after_tm_s["p5"],
                "P50 After-tax Full TCO ($/ton-mile)": after_tm_s["p50"],
                "P95 After-tax Full TCO ($/ton-mile)": after_tm_s["p95"],
            })
        rows.append(row)
        all_full_rows.append(row)

        c_row = {
            "Application": res["label"],
            "Economic inputs": app_mode_label,
            "Mileage method": res.get("mileage_mode", MILEAGE_MODE_CONSTANT),
            "Vehicle": vt.upper(),
        }
        for cname in FULL_TCO_COMPONENTS:
            c_arr = np.array(res["full_tco_component_mile"][vt][cname], dtype=float)
            c_total_arr = np.array(res["full_tco_component_total_pv"][vt][cname], dtype=float)
            c_row[f"P50 {FULL_TCO_LABELS[cname]} ($/mile)"] = np.percentile(c_arr, 50)
            c_row[f"P50 {FULL_TCO_LABELS[cname]} Total PV ($)"] = np.percentile(c_total_arr, 50)
        c_row["P50 Federal tax benefit ($/mile)"] = tax_mile_s["p50"]
        c_row["P50 Federal tax benefit Total PV ($)"] = tax_total_s["p50"]
        c_row["P50 Pre-tax Full TCO ($/mile)"] = mile_s["p50"]
        c_row["P50 After-tax Full TCO ($/mile)"] = after_mile_s["p50"]
        c_row["P50 Pre-tax Total PV TCO ($)"] = total_s["p50"]
        c_row["P50 After-tax Total PV TCO ($)"] = after_total_s["p50"]
        component_rows.append(c_row)
        all_component_rows.append(c_row)

    st.markdown(f"### {res['label']} — {app_mode_label}")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
        with st.expander(f"{res['label']} — P50 component breakdown ($/mile)", expanded=False):
            st.dataframe(pd.DataFrame(component_rows), width="stretch")
    else:
        st.info("No selected vehicles available for this application.")

full_summary_df = pd.DataFrame(all_full_rows)
component_summary_df = pd.DataFrame(all_component_rows)

if not full_summary_df.empty:
    st.download_button(
        label="Download Full TCO summary as CSV",
        data=full_summary_df.to_csv(index=False).encode("utf-8"),
        file_name="full_tco_summary.csv",
        mime="text/csv",
    )

if not component_summary_df.empty:
    st.download_button(
        label="Download Full TCO component breakdown as CSV",
        data=component_summary_df.to_csv(index=False).encode("utf-8"),
        file_name="full_tco_component_breakdown.csv",
        mime="text/csv",
    )

# ============================================================
# INDIVIDUAL STACKED FULL TCO PLOTS
# ============================================================
def make_full_tco_stacked_plot_for_app(app_key: str, metric: str, tax_view: str = "pre_tax"):
    """Create one stacked Full TCO plot for one application only.

    tax_view = "pre_tax" shows current Full TCO.
    tax_view = "after_tax" shows the same positive cost stack plus a negative
    Federal tax benefit segment; the black error bar is the after-tax total.
    """
    if metric == "$/mile":
        comp_key = "full_tco_component_mile"
        if tax_view == "after_tax":
            total_key = "after_tax_full_tco_mile"
            tax_key = "federal_tax_benefit_mile"
            ylabel = "After-tax Full TCO ($/mile)"
            file_label = "after_tax_mile"
            plot_label = "After-tax Full TCO"
        else:
            total_key = "full_tco_mile"
            tax_key = None
            ylabel = "Pre-tax Full TCO ($/mile)"
            file_label = "pre_tax_mile"
            plot_label = "Pre-tax Full TCO"
    elif metric == "Total PV TCO ($)":
        comp_key = "full_tco_component_total_pv"
        if tax_view == "after_tax":
            total_key = "after_tax_full_tco_total_pv"
            tax_key = "federal_tax_benefit_total_pv"
            ylabel = "After-tax total PV TCO ($)"
            file_label = "after_tax_total_pv"
            plot_label = "After-tax Total PV TCO"
        else:
            total_key = "full_tco_total_pv"
            tax_key = None
            ylabel = "Pre-tax total PV TCO ($)"
            file_label = "pre_tax_total_pv"
            plot_label = "Pre-tax Total PV TCO"
    else:
        comp_key = "full_tco_component_tm"
        if tax_view == "after_tax":
            total_key = "after_tax_full_tco_tm"
            tax_key = "federal_tax_benefit_tm"
            ylabel = "After-tax Full TCO ($/ton-mile)"
            file_label = "after_tax_ton_mile"
            plot_label = "After-tax Full TCO"
        else:
            total_key = "full_tco_tm"
            tax_key = None
            ylabel = "Pre-tax Full TCO ($/ton-mile)"
            file_label = "pre_tax_ton_mile"
            plot_label = "Pre-tax Full TCO"

    res = all_results[app_key]
    vehs = [vt for vt in res["VEH_LIST"] if vt in selected_vehicles_display]
    mode_label = econ_input_set_label(res["economic_mode"])
    mileage_label = res.get("mileage_mode", MILEAGE_MODE_CONSTANT)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    if len(vehs) == 0:
        ax.axis("off")
        ax.set_title(f"{res['label']}\nNo selected vehicles")
        return fig, file_label, mode_label, plot_label

    x = np.arange(len(vehs))
    bottoms = np.zeros(len(vehs))

    total_p5 = []
    total_p50 = []
    total_p95 = []

    for vt in vehs:
        arr = np.array(res[total_key][vt], dtype=float)
        total_p5.append(np.percentile(arr, 5))
        total_p50.append(np.percentile(arr, 50))
        total_p95.append(np.percentile(arr, 95))

    total_p5 = np.array(total_p5)
    total_p50 = np.array(total_p50)
    total_p95 = np.array(total_p95)

    for cname in FULL_TCO_COMPONENTS:
        vals = []
        for vt in vehs:
            c_arr = np.array(res[comp_key][vt][cname], dtype=float)
            vals.append(np.percentile(c_arr, 50))
        vals = np.array(vals)

        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=FULL_TCO_LABELS[cname],
            color=FULL_TCO_COLORS[cname],
            edgecolor="black",
            linewidth=0.7,
            width=0.58,
            alpha=1.0,
        )
        bottoms += vals

    if tax_view == "after_tax" and tax_key is not None:
        tax_vals = []
        for vt in vehs:
            tax_arr = np.array(res[tax_key][vt], dtype=float)
            tax_vals.append(np.percentile(tax_arr, 50))
        tax_vals = np.array(tax_vals)

        ax.bar(
            x,
            -tax_vals,
            bottom=bottoms,
            label=FULL_TCO_LABELS["federal_tax_benefit"],
            color=FULL_TCO_COLORS["federal_tax_benefit"],
            edgecolor="black",
            linewidth=0.7,
            width=0.58,
            alpha=1.0,
            hatch="//",
        )

    ax.errorbar(
        x,
        total_p50,
        yerr=[total_p50 - total_p5, total_p95 - total_p50],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=10,
    )
    ax.scatter(x, total_p50, color="black", s=30, zorder=11)

    ax.set_xticks(x)
    ax.set_xticklabels([_vehicle_pretty(v) for v in vehs])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{res['label']}\n{plot_label} — {mode_label}\nMileage: {mileage_label}")
    ax.grid(axis="y", alpha=0.3)

    if metric == "Total PV TCO ($)":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_dollar_axis_millions))
        ax.yaxis.get_offset_text().set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    return fig, file_label, mode_label, plot_label


st.subheader("Full TCO Component Stacked Plots")
st.markdown(
    "The app now plots only the active application case to reduce Streamlit Cloud load. "
    "You can still select more than one plot metric in the sidebar. "
    "$/ton-mile plots are shown only for Drayage and Long Haul. "
    "The after-tax plot keeps the same positive cost stack and adds a hatched negative Federal tax benefit segment. "
    "The black error bar shows the selected total P5–P95."
)

plot_views = []
if tax_plot_display in ["Show both", "Pre-tax only"]:
    plot_views.append("pre_tax")
if tax_plot_display in ["Show both", "After-tax only"]:
    plot_views.append("after_tax")

if len(metric_display_list) == 0:
    st.warning("No Full TCO plot metric selected. Select at least one metric in the sidebar.")
else:
    for app_key in selected_apps:
        res = all_results[app_key]

        for metric_display in metric_display_list:
            if metric_display == "$/ton-mile" and app_key not in FREIGHT_TON_MILE_APPS:
                st.info(
                    f"Skipping {res['label']} for $/ton-mile plots. "
                    "Ton-mile plots are shown only for Drayage and Long Haul."
                )
                continue

            for tax_view in plot_views:
                fig_app, file_label, mode_label, plot_label = make_full_tco_stacked_plot_for_app(
                    app_key,
                    metric_display,
                    tax_view=tax_view,
                )

                st.markdown(f"### {res['label']} — {plot_label} — {mode_label}")
                st.pyplot(fig_app)

                buf_app = BytesIO()
                fig_app.savefig(buf_app, format="png", dpi=600, bbox_inches="tight")
                buf_app.seek(0)

                st.download_button(
                    label=f"Download {res['label']} {plot_label} stacked plot PNG, 600 dpi ({metric_display})",
                    data=buf_app,
                    file_name=f"full_tco_stacked_{app_key}_{mode_label.lower()}_{file_label}.png",
                    mime="image/png",
                    key=f"download_full_tco_plot_{app_key}_{file_label}_{mode_label}_{tax_view}_{metric_display}",
                )

                plt.close(fig_app)

# ============================================================
# ORIGINAL LCOD TABLE, KEPT FOR BREAKEVEN TRANSPARENCY ONLY
# ============================================================
with st.expander("Original LCOD summary used for breakeven only", expanded=False):
    old_rows = []
    for app_key in selected_apps:
        res = all_results[app_key]
        for vt in res["VEH_LIST"]:
            if vt not in selected_vehicles_display:
                continue
            mile_s = _summarize_percentiles(np.array(res["lcod_mile"][vt], dtype=float), PCTILES)
            tm_s = _summarize_percentiles(np.array(res["lcod_tm"][vt], dtype=float), PCTILES)
            old_rows.append({
                "Application": res["label"],
                "Vehicle": vt.upper(),
                "P5 original LCOD ($/mile)": mile_s["p5"],
                "P50 original LCOD ($/mile)": mile_s["p50"],
                "P95 original LCOD ($/mile)": mile_s["p95"],
                "P5 original LCOD ($/ton-mile)": tm_s["p5"],
                "P50 original LCOD ($/ton-mile)": tm_s["p50"],
                "P95 original LCOD ($/ton-mile)": tm_s["p95"],
            })
    if old_rows:
        st.dataframe(pd.DataFrame(old_rows), width="stretch")
    else:
        st.info("No original LCOD rows to display.")

# ============================================================
# BREAKEVEN ANALYSIS, ORIGINAL LCOD ONLY, NO PROBABILITY CURVE
# ============================================================
st.subheader("Breakeven Analysis vs Diesel")

st.markdown(
    "Breakeven purchase price is calculated using the original LCOD method from the old app. "
    "The Full TCO components are not used in this breakeven calculation."
)

available_be_apps = [
    app_key for app_key in selected_apps
    if "diesel" in all_results[app_key]["VEH_LIST"]
]

if len(available_be_apps) == 0:
    st.warning("Diesel must be available to calculate breakeven.")
else:
    be_col1, be_col2 = st.columns(2)

    with be_col1:
        be_app = st.selectbox(
            "Select application for breakeven",
            options=available_be_apps,
            format_func=lambda x: APPLICATIONS[x]["label"]
        )

    available_alt = [
        vt for vt in all_results[be_app]["VEH_LIST"]
        if vt != "diesel"
    ]

    with be_col2:
        be_vehicle = st.selectbox(
            "Select alternative vehicle",
            options=available_alt,
            format_func=lambda x: x.upper()
        )

    res = all_results[be_app]

    diesel_lcod = np.array(res["lcod_mile"]["diesel"], dtype=float)

    alt_fuel = np.array(res["breakeven_inputs"][be_vehicle]["fuel_per_mile"], dtype=float)
    alt_maint = np.array(res["breakeven_inputs"][be_vehicle]["maintenance_per_mile"], dtype=float)
    alt_lifetime_miles = np.array(res["breakeven_inputs"][be_vehicle]["lifetime_miles"], dtype=float)
    alt_residual = np.array(res["breakeven_inputs"][be_vehicle]["residual_cost"], dtype=float)
    alt_current_purchase = np.array(res["breakeven_inputs"][be_vehicle]["purchase_total"], dtype=float)

    purchase_breakeven = (
        (diesel_lcod - alt_fuel - alt_maint) * alt_lifetime_miles
        + alt_residual
    )

    valid_mask = np.isfinite(purchase_breakeven)
    purchase_breakeven = purchase_breakeven[valid_mask]
    alt_current_purchase = alt_current_purchase[valid_mask]

    be_p5 = np.percentile(purchase_breakeven, 5)
    be_p50 = np.percentile(purchase_breakeven, 50)
    be_p95 = np.percentile(purchase_breakeven, 95)

    current_p5 = np.percentile(alt_current_purchase, 5)
    current_p50 = np.percentile(alt_current_purchase, 50)
    current_p95 = np.percentile(alt_current_purchase, 95)

    gap_p50 = current_p50 - be_p50

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Breakeven P50 purchase price", f"${be_p50:,.0f}")
    c2.metric("Current P50 purchase price", f"${current_p50:,.0f}")
    c3.metric("P50 gap: Current - Breakeven", f"${gap_p50:,.0f}")
    c4.metric("Breakeven P5–P95", f"${be_p5:,.0f}–${be_p95:,.0f}")

    be_table = pd.DataFrame([
        {
            "Application": res["label"],
            "Vehicle": be_vehicle.upper(),
            "Breakeven P5 ($)": be_p5,
            "Breakeven P50 ($)": be_p50,
            "Breakeven P95 ($)": be_p95,
            "Current Purchase P5 ($)": current_p5,
            "Current Purchase P50 ($)": current_p50,
            "Current Purchase P95 ($)": current_p95,
            "P50 Gap: Current - Breakeven ($)": gap_p50,
        }
    ])

    st.markdown("### Breakeven Summary Table")
    st.dataframe(be_table, width="stretch")

    st.download_button(
        label="Download breakeven summary as CSV",
        data=be_table.to_csv(index=False).encode("utf-8"),
        file_name=f"breakeven_{be_app}_{be_vehicle}.csv",
        mime="text/csv",
    )

# ============================================================
# FINAL NOTE
# ============================================================
st.info(
    "Existing LCOD app inputs are preserved. Added economic and federal corporate tax-shield inputs are separate and can be Default or Customized for each application. "
    "The app now reports and plots both pre-tax Full TCO and after-tax Full TCO. "
    "Breakeven uses the original LCOD formula only. Drayage and Long Haul CNG still use diesel-like trial parameters from the old app unless you replace them."
)
