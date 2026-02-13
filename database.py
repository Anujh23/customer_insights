"""
PostgreSQL Database Module for Customer 360 Insight
Product-specific tables: {product}_disbursed and {product}_collection
"""
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

GRACE_PERIOD_DAYS = 3


@contextmanager
def get_db_connection():
    """Context manager for PostgreSQL database connections."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_table_name(product: str, table_type: str) -> str:
    """Generate table name for product-specific table."""
    return f"{product.lower()}_{table_type}"


def sanitize_column_name(col: str) -> str:
    """Sanitize column name for PostgreSQL."""
    return col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')[:60]


def create_product_tables(product: str, disbursed_columns: List[str], collection_columns: List[str]) -> None:
    """Create disbursed and collection tables for a specific product with all CSV columns."""
    disbursed_table = get_table_name(product, 'disbursed')
    collection_table = get_table_name(product, 'collection')

    def build_columns(col_list: List[str]) -> List[str]:
        cols = ["id SERIAL PRIMARY KEY"]
        for col in col_list:
            col_clean = sanitize_column_name(col)
            if col_clean.lower() != 'id':
                cols.append(f'"{col_clean}" VARCHAR(500)')
        cols.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        return cols

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            disbursed_cols = build_columns(disbursed_columns)
            collection_cols = build_columns(collection_columns)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {disbursed_table} (
                    {', '.join(disbursed_cols)}
                )
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {collection_table} (
                    {', '.join(collection_cols)}
                )
            """)

            # Add unique constraint for duplicate prevention
            try:
                cur.execute(f"""
                    ALTER TABLE {disbursed_table} 
                    ADD CONSTRAINT {disbursed_table}_lead_loan_unique 
                    UNIQUE ("LeadID", "Loan_No")
                """)
            except psycopg2.Error:
                conn.rollback()  # Constraint already exists

            try:
                cur.execute(f"""
                    ALTER TABLE {collection_table} 
                    ADD CONSTRAINT {collection_table}_lead_loan_unique 
                    UNIQUE ("LeadID", "Loan_No")
                """)
            except psycopg2.Error:
                conn.rollback()  # Constraint already exists

            conn.commit()
            logger.info(f"Created tables: {disbursed_table}, {collection_table}")


def insert_dataframe(cur, df: pd.DataFrame, table_name: str) -> tuple:
    """Insert DataFrame records into a PostgreSQL table with upsert (update on duplicate)."""
    if df.empty:
        return 0, 0

    columns = []
    for col in df.columns:
        col_clean = sanitize_column_name(col)
        if col_clean.lower() != 'id':
            columns.append(f'"{col_clean}"')

    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)

    # Build ON CONFLICT update clause (exclude LeadID and Loan_No from update)
    update_cols = [col for col in columns if col not in ['"LeadID"', '"Loan_No"']]
    update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_cols])

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            if sanitize_column_name(col).lower() != 'id':
                val = row.get(col, '')
                values.append(None if pd.isna(val) else str(val).strip())

        try:
            cur.execute(f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT ("LeadID", "Loan_No") DO UPDATE SET
                    {update_clause}
            """, values)
            # Check if it was insert or update
            if cur.rowcount == 1:
                # Could be either, check if row exists logic at DB level
                inserted += 1
        except psycopg2.Error as e:
            logger.warning(f"Error inserting row: {e}")
            continue

    return inserted, len(df) - inserted


def process_uploaded_files_pg(disbursed_df: pd.DataFrame, collection_df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """Process uploaded CSV files and insert into product-specific PostgreSQL tables."""
    disbursed_df.columns = disbursed_df.columns.str.strip()
    collection_df.columns = collection_df.columns.str.strip()

    create_product_tables(product, list(disbursed_df.columns), list(collection_df.columns))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            disbursed_table = get_table_name(product, 'disbursed')
            collection_table = get_table_name(product, 'collection')

            disbursed_inserted, disbursed_updated = insert_dataframe(cur, disbursed_df, disbursed_table)
            collection_inserted, collection_updated = insert_dataframe(cur, collection_df, collection_table)

            conn.commit()
            logger.info(f"{product}: {disbursed_inserted} inserted/{disbursed_updated} updated disbursed, {collection_inserted} inserted/{collection_updated} updated collection")

    return {
        'success': True,
        'product': product,
        'disbursed_inserted': disbursed_inserted,
        'disbursed_updated': disbursed_updated,
        'collection_inserted': collection_inserted,
        'collection_updated': collection_updated
    }


def list_products_pg() -> List[str]:
    """Get list of products by finding all tables ending with _disbursed."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%_disbursed'
                ORDER BY table_name
            """)
            tables = cur.fetchall()
            return [row['table_name'].replace('_disbursed', '').upper() for row in tables]


def determine_payment_status(collected_date: Optional[datetime], repay_date: Optional[datetime]) -> str:
    """Determine payment status based on collection and repayment dates."""
    if not collected_date or not repay_date:
        return "NOT_COLLECTED"
    
    if isinstance(collected_date, str):
        collected_date = datetime.strptime(collected_date, '%Y-%m-%d').date()
    if isinstance(repay_date, str):
        repay_date = datetime.strptime(repay_date, '%Y-%m-%d').date()
    
    if hasattr(collected_date, 'date'):
        collected_date = collected_date.date()
    if hasattr(repay_date, 'date'):
        repay_date = repay_date.date()
    
    if collected_date < repay_date:
        return "EARLY"
    elif collected_date == repay_date:
        return "ON_TIME"
    elif (collected_date - repay_date).days <= GRACE_PERIOD_DAYS:
        return "GRACE_PERIOD"
    else:
        return "LATE"


def search_pan_pg(pan: str, case_sensitive: bool = False) -> Dict[str, Any]:
    """Search PAN across all product tables in PostgreSQL."""
    products = list_products_pg()
    all_results = []

    with get_db_connection() as conn:
        for product in products:
            disbursed_table = get_table_name(product, 'disbursed')
            collection_table = get_table_name(product, 'collection')

            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT d.*, c."Collected_Date", c."Collected_Amount"
                        FROM {disbursed_table} d
                        LEFT JOIN {collection_table} c 
                            ON d."LeadID" = c."LeadID" AND d."Loan_No" = c."Loan_No"
                        WHERE LOWER(d."Pancard") = LOWER(%s)
                        ORDER BY d."Repay_Date"
                    """, (pan,))

                    for row in cur.fetchall():
                        result = dict(row)
                        result['PaymentStatus'] = determine_payment_status(
                            result.get('Collected_Date'),
                            result.get('Repay_Date')
                        )
                        result['Product'] = product.upper()
                        all_results.append(result)

            except psycopg2.Error as e:
                logger.warning(f"Error querying {product}: {e}")
                continue

    return {
        'success': True,
        'pan': pan,
        'total_records': len(all_results),
        'records': all_results
    }


def run_sql_query_pg(query: str) -> Dict[str, Any]:
    """Execute SQL query on PostgreSQL database - SELECT only."""
    # Normalize query for validation
    normalized = query.strip().upper()
    
    # Block dangerous commands
    forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'TRUNCATE', 'ALTER', 'GRANT', 'REVOKE']
    for keyword in forbidden_keywords:
        if keyword in normalized:
            raise ValueError(f"{keyword} commands are not allowed. Only SELECT queries are permitted.")
    
    # Ensure query starts with SELECT
    if not normalized.startswith('SELECT'):
        raise ValueError("Only SELECT queries are allowed.")
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            
            if cur.description:
                records = cur.fetchall()
                return {
                    'success': True,
                    'total_records': len(records),
                    'records': [dict(row) for row in records]
                }
            else:
                return {
                    'success': True,
                    'total_records': 0,
                    'records': [],
                    'message': 'Query executed successfully'
                }
