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
from psycopg2.extras import RealDictCursor, execute_values
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


def normalize_upload_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column headers coming from CSVs so downstream SQL, constraints,
    # and joins can rely on stable names.
    df = df.copy()
    df.columns = df.columns.str.strip()

    def _key(col: str) -> str:
        return sanitize_column_name(col).lower()

    canonical_map = {
        # disbursed + collection common keys
        'leadid': 'LeadID',
        'lead_id': 'LeadID',
        'loan_no': 'Loan_No',
        'loanno': 'Loan_No',
        # disbursed
        'pancard': 'Pancard',
        'pan': 'Pancard',
        'repay_date': 'Repay_Date',
        'repayment_date': 'Repay_Date',
        'loan_amount': 'LoanAmount',
        'disbursed_amount': 'LoanAmount',
        'amount': 'LoanAmount',
        'sanctioned_amount': 'LoanAmount',
        # collection
        'collected_date': 'Collected_Date',
        'collection_date': 'Collected_Date',
        'collected_amount': 'Collected_Amount',
        'collection_amount': 'Collected_Amount',
    }

    rename_map = {}
    for col in df.columns:
        k = _key(col)
        if k in canonical_map:
            rename_map[col] = canonical_map[k]

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


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

            # Commit table creation first so later rollbacks don't remove the tables.
            conn.commit()

            # Add unique constraint for duplicate prevention
            try:
                cur.execute(f"""
                    ALTER TABLE {disbursed_table} 
                    ADD CONSTRAINT {disbursed_table}_lead_loan_unique 
                    UNIQUE ("LeadID", "Loan_No")
                """)
                conn.commit()
            except psycopg2.Error:
                conn.rollback()  # Constraint already exists

            try:
                cur.execute(f"""
                    ALTER TABLE {collection_table} 
                    ADD CONSTRAINT {collection_table}_lead_loan_unique 
                    UNIQUE ("LeadID", "Loan_No")
                """)
                conn.commit()
            except psycopg2.Error:
                conn.rollback()  # Constraint already exists
            logger.info(f"Created tables: {disbursed_table}, {collection_table}")


def insert_dataframe(cur, df: pd.DataFrame, table_name: str) -> tuple:
    """Insert DataFrame records into a PostgreSQL table with bulk upsert (update on duplicate)."""
    if df.empty:
        return 0, 0

    # Normalize columns for insertion
    columns = []
    col_index_map = {}  # Map column name to its index in values
    for idx, col in enumerate(df.columns):
        col_clean = sanitize_column_name(col)
        if col_clean.lower() != 'id':
            columns.append(f'"{col_clean}"')
            col_index_map[col_clean] = len(columns) - 1

    if not columns:
        return 0, 0

    # Build ON CONFLICT update clause (exclude LeadID and Loan_No from update)
    update_cols = [col for col in columns if col not in ['"LeadID"', '"Loan_No"']]
    update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_cols])

    # Prepare data as list of tuples for bulk insert
    data_tuples = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            if sanitize_column_name(col).lower() != 'id':
                val = row.get(col, '')
                values.append(None if pd.isna(val) else str(val).strip())
        data_tuples.append(tuple(values))

    inserted_count = 0
    updated_count = 0
    batch_size = 1000  # Process in batches of 1000 rows
    
    # Get indices for LeadID and Loan_No for deduplication
    leadid_idx = col_index_map.get('LeadID')
    loan_no_idx = col_index_map.get('Loan_No')
    
    logger.info(f"Inserting {len(data_tuples)} rows into {table_name} with columns: {columns}")

    try:
        # Process in batches for better performance and error handling
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            
            # Deduplicate within batch: keep last occurrence of each (LeadID, Loan_No)
            if leadid_idx is not None and loan_no_idx is not None:
                seen = {}
                deduplicated = []
                for idx, row in enumerate(batch):
                    key = (row[leadid_idx], row[loan_no_idx])
                    seen[key] = idx
                # Keep only the last occurrence of each key
                keep_indices = set(seen.values())
                deduplicated = [batch[idx] for idx in sorted(keep_indices)]
                if len(deduplicated) < len(batch):
                    logger.info(f"Deduplicated batch from {len(batch)} to {len(deduplicated)} rows")
                batch = deduplicated
            
            if not batch:
                continue
            
            # Use execute_values for bulk insert with ON CONFLICT
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES %s
                ON CONFLICT ("LeadID", "Loan_No") DO UPDATE SET
                    {update_clause}
            """
            
            try:
                execute_values(cur, query, batch, page_size=len(batch))
                inserted_count += len(batch)
                logger.info(f"Batch {i//batch_size + 1}: Inserted/updated {len(batch)} rows")
            except psycopg2.Error as e:
                error_msg = str(e)
                logger.warning(f"Batch insert error at batch {i//batch_size + 1}: {error_msg}")
                
                # If it's a duplicate key error, fall back to row-by-row
                if "ON CONFLICT" in error_msg or "duplicate" in error_msg.lower():
                    logger.info(f"Falling back to row-by-row insert for batch {i//batch_size + 1}")
                    for row_idx, row_values in enumerate(batch):
                        try:
                            cur.execute(f"""
                                INSERT INTO {table_name} ({', '.join(columns)})
                                VALUES ({', '.join(['%s'] * len(columns))})
                                ON CONFLICT ("LeadID", "Loan_No") DO UPDATE SET
                                    {update_clause}
                            """, row_values)
                            inserted_count += 1
                        except psycopg2.Error as row_e:
                            logger.debug(f"Row {i + row_idx} failed: {row_e}")
                            continue
                else:
                    # Re-raise if it's a different error
                    raise

    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        raise

    logger.info(f"Completed: {inserted_count} rows processed for {table_name}")
    return inserted_count, 0


def process_uploaded_files_pg(disbursed_df: pd.DataFrame, collection_df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """Process uploaded CSV files and insert into product-specific PostgreSQL tables."""
    disbursed_df = normalize_upload_columns(disbursed_df)
    collection_df = normalize_upload_columns(collection_df)

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


def get_table_columns(table_name: str) -> List[Dict[str, Any]]:
    """Get all columns for a specific table."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cur.fetchall()
            return [
                {
                    'column_name': row['column_name'],
                    'data_type': row['data_type'],
                    'is_nullable': row['is_nullable'],
                    'quoted_name': f'"{row["column_name"]}"'
                }
                for row in columns
            ]


def parse_date_flexible(date_val):
    """Parse date string in multiple formats (DD-MM-YYYY or YYYY-MM-DD)."""
    if not date_val:
        return None
    
    if isinstance(date_val, datetime):
        return date_val.date() if hasattr(date_val, 'date') else date_val
    
    date_str = str(date_val).strip()
    if not date_str:
        return None
    
    # Try DD-MM-YYYY format first (common in Indian data)
    try:
        return datetime.strptime(date_str, '%d-%m-%Y').date()
    except ValueError:
        pass
    
    # Try YYYY-MM-DD format
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        pass
    
    # Try DD/MM/YYYY format
    try:
        return datetime.strptime(date_str, '%d/%m/%Y').date()
    except ValueError:
        pass
    
    # If all fail, log and return None
    logger.warning(f"Could not parse date: {date_str}")
    return None


def determine_payment_status(collected_date: Optional[datetime], repay_date: Optional[datetime]) -> str:
    """Determine payment status based on collection and repayment dates."""
    # Parse dates flexibly
    collected_date = parse_date_flexible(collected_date)
    repay_date = parse_date_flexible(repay_date)
    
    if not collected_date or not repay_date:
        return "NOT_COLLECTED"
    
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
                        # Convert underscore column names to camelCase for frontend
                        result['RepayDate'] = result.get('Repay_Date')
                        result['CollectionDate'] = result.get('Collected_Date')
                        result['LoanAmount'] = result.get('LoanAmount') or result.get('Loan_Amount')
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
