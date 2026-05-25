"""
RetailPulse — Stage 1 ETL
Reads Online Retail II Excel, cleans it, loads into MySQL.
Run once: python etl_load.py
"""

import pandas as pd
from sqlalchemy import create_engine, text

# ── CONFIG ──────────────────────────────────────────────────
DB_USER     = "root"          # change to your MySQL username
DB_PASSWORD = "aagam"  # change to your MySQL password
DB_HOST     = "localhost"
DB_NAME     = "retailpulse"
#EXCEL_PATH  = "data\online_retail_II.csv"
CSV_PATH    = "data/online_retail_II.csv"
# ────────────────────────────────────────────────────────────

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)


'''def load_raw():
    """Load both sheets from the Excel file and combine."""
    print("Reading Excel (this takes ~30 seconds for 1M rows)...")
    df1 = pd.read_excel(EXCEL_PATH, sheet_name="Year 2009-2010", dtype=str)
    df2 = pd.read_excel(EXCEL_PATH, sheet_name="Year 2010-2011", dtype=str)
    df  = pd.concat([df1, df2], ignore_index=True)
    print(f"  Raw rows loaded: {len(df):,}")
    return df'''

def load_raw():
    """Load CSV dataset."""
    print("Reading CSV file...")

    df = pd.read_csv(
        CSV_PATH,
        encoding="ISO-8859-1",
        dtype=str
    )

    print(f"  Raw rows loaded: {len(df):,}")
    return df

def clean(df):
    """All cleaning decisions documented here."""
    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Rename to match schema
    '''df = df.rename(columns={
        "invoice":      "invoice_no",
        "stockcode":    "stock_code",
        "customer_id":  "customer_id",
        "price":        "unit_price",
    })'''
    df = df.rename(columns={
    "invoice":        "invoice_no",
    "stockcode":      "stock_code",
    "customer_id":    "customer_id",
    "price":          "unit_price",
    # ADD these two lines to catch both possible formats:
    "invoicedate":    "invoice_date",
    "invoice date":   "invoice_date",
})
    # Drop rows missing customer_id — can't do RFM without it
    before = len(df)
    df = df.dropna(subset=["customer_id"])
    print(f"  Dropped {before - len(df):,} rows with no customer_id")

    # Drop cancellations (invoice starts with 'C')
    before = len(df)
    df = df[~df["invoice_no"].str.startswith("C", na=False)]
    print(f"  Dropped {before - len(df):,} cancelled invoices")

    # Drop bad quantities and prices
    df["quantity"]   = pd.to_numeric(df["quantity"],   errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    before = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    print(f"  Dropped {before - len(df):,} rows with negative/zero qty or price")

    # Parse invoice date
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    # Clean types
    df["customer_id"] = df["customer_id"].astype(float).astype(int)
    df["description"] = df["description"].fillna("Unknown").str.strip()
    df["country"]     = df["country"].str.strip()

    print(f"  Clean rows remaining: {len(df):,}")
    return df


def load_to_mysql(df):
    """Insert into normalized tables in correct FK order."""
    print("\nLoading into MySQL...")

    # 1. Customers
    customers = (
        df[["customer_id", "country"]]
        .drop_duplicates("customer_id")
        .reset_index(drop=True)
    )
    customers.to_sql("customers", engine, if_exists="append", index=False)
    print(f"  customers: {len(customers):,} rows")

    # 2. Products
    products = (
        df[["stock_code", "description"]]
        .drop_duplicates("stock_code")
        .reset_index(drop=True)
    )
    products.to_sql("products", engine, if_exists="append", index=False)
    print(f"  products:  {len(products):,} rows")

    # 3. Invoices
    invoices = (
        df[["invoice_no", "invoice_date", "customer_id"]]
        .drop_duplicates("invoice_no")
        .reset_index(drop=True)
    )
    invoices.to_sql("invoices", engine, if_exists="append", index=False)
    print(f"  invoices:  {len(invoices):,} rows")

    # 4. Invoice items (all line rows)
    items = df[["invoice_no", "stock_code", "quantity", "unit_price"]].copy()
    items.to_sql("invoice_items", engine, if_exists="append",
                 index=False, chunksize=5000)
    print(f"  items:     {len(items):,} rows")


def verify():
    """Quick sanity check after load."""
    print("\nVerification queries:")
    checks = {
        "total customers":     "SELECT COUNT(*) FROM customers",
        "total invoices":      "SELECT COUNT(*) FROM invoices",
        "total line items":    "SELECT COUNT(*) FROM invoice_items",
        "total revenue":       "SELECT ROUND(SUM(quantity*unit_price),2) FROM invoice_items",
        "date range":          "SELECT MIN(invoice_date), MAX(invoice_date) FROM invoices",
    }
    with engine.connect() as conn:
        for label, sql in checks.items():
            result = conn.execute(text(sql)).fetchone()
            print(f"  {label}: {result[0] if len(result)==1 else result}")


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    load_to_mysql(df)
    verify()
    print("\nStage 1 complete. Database is live.")