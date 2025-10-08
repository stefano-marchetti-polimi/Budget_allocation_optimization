# 1) Install
# pip install hydromt-fiat

try:
    from hydromt_fiat.fiat import FiatModel as _FiatModel
except ImportError:
    from hydromt_fiat.fiat import FIATModel as _FiatModel
import hydromt_fiat, hydromt
print("hydromt_fiat version:", hydromt_fiat.__version__, "| hydromt version:", hydromt.__version__)
print("Using model class:", _FiatModel.__name__)
from pathlib import Path
import pandas as pd

root = Path("fiat_hazus_demo")
m = _FiatModel(root=root, mode="w")
m.setup_config(                           # this tells FIAT which internal data catalog to use
    catalog_fn="hydromt_fiat_catalog_USA.yml"
)

# 3) Load the HAZUS curves + linking table (required)
#    NOTE: unit must be FEET for HAZUS vulnerability (per docs)
if hasattr(m.__class__, "setup_vulnerability"):
    m.setup_vulnerability(
        vulnerability_fn="default_vulnerability_curves",
        vulnerability_identifiers_and_linking_fn="default_hazus_iwr_linking",
        unit="feet",
    )
elif hasattr(m.__class__, "setup_vulnerability_from_csv"):
    # Fall back to CSV-based setup if available
    vuln_dir = root / "vulnerability"
    vuln_csv = vuln_dir / "vulnerability.csv"
    if not vuln_csv.exists():
        raise RuntimeError(
            "No 'setup_vulnerability' method and no vulnerability CSV found. "
            "Upgrade hydromt-fiat or generate 'vulnerability.csv' manually."
        )
    m.setup_vulnerability_from_csv(csv_fn=str(vuln_csv), unit="feet")
elif hasattr(m.__class__, "copy_vulnerability_tables"):
    # Very old API: copy bundled vulnerability tables (signature may vary across versions)
    try:
        m.copy_vulnerability_tables()
    except TypeError:
        # Some versions expect a source key like 'hazus'
        m.copy_vulnerability_tables(source="hazus")
else:
    # Fallback for very old/dev builds: try to locate packaged HAZUS CSVs and load them directly
    import sys
    from pathlib import Path as _Path
    import pandas as _pd
    pkg_dir = _Path(hydromt_fiat.__file__).parent
    # Search for packaged CSVs that look like vulnerability and linking tables
    candidate_csvs = list(pkg_dir.glob("**/*.csv"))
    vuln_csvs = [p for p in candidate_csvs if "vuln" in p.name.lower() or "vulnerability" in p.name.lower()]
    link_csvs = [p for p in candidate_csvs if "link" in p.name.lower() and "hazus" in p.name.lower()]
    if not vuln_csvs:
        raise AttributeError(
            "No vulnerability setup method and no packaged vulnerability CSVs found in hydromt_fiat. "
            "Please upgrade to a stable release (e.g., >=0.5.7) which exposes FiatModel.setup_vulnerability."
        )
    # Pick the first match; advanced users can swap if needed
    _vuln_csv = str(vuln_csvs[0])
    print(f"[Fallback] Loading vulnerability CSV: {_vuln_csv}")
    df_vuln = _pd.read_csv(_vuln_csv, header=None)
    # Some packaged CSVs have header rows; try intelligent header handling
    if not df_vuln.iloc[0,0].dtype == object:
        df_vuln = _pd.read_csv(_vuln_csv)
    m.set_tables(df_vuln, name="vulnerability")
    # Optional: linking table if present
    if link_csvs:
        _link_csv = str(link_csvs[0])
        print(f"[Fallback] Loading HAZUS linking CSV: {_link_csv}")
        df_link = _pd.read_csv(_link_csv)
        m.set_tables(df_link, name="hazus_linking")
    print("[Fallback] Vulnerability tables set from packaged CSVs. Consider upgrading hydromt_fiat to use setup_vulnerability().")

# 4) Write files to disk so you can inspect them
m.write()

# 5) Read the generated CSVs
vuln_dir = root / "vulnerability"
print("Files in vulnerability folder:", list(vuln_dir.glob("*.csv")))

# Typical filenames:
vuln_csv   = vuln_dir / "vulnerability.csv"          # the curves database in Delft-FIAT CSV format
link_csv   = vuln_dir / "hazus_linking.csv"          # maps HAZUS classes to curve ids (name may vary)
if vuln_csv.exists():
    df_curves = pd.read_csv(vuln_csv)
    print(df_curves.head())

# You can now filter by curve name (e.g., 'Electric substation – structure', etc.)
