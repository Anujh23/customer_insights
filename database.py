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
        'name': 'Name',
        'customername': 'Name',
        'customer_name': 'Name',
        'fullname': 'Name',
        'full_name': 'Name',
        'applicantname': 'Name',
        'applicant_name': 'Name',
        'mobile': 'Mobile',
        'phone': 'Mobile',
        'mobileno': 'Mobile',
        'mobile_no': 'Mobile',
        'phonenumber': 'Mobile',
        'phone_number': 'Mobile',
        'contact': 'Mobile',
        'contactno': 'Mobile',
        'contact_no': 'Mobile',
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


def search_pan_pg(pan: str = None, name: str = None, mobile: str = None, case_sensitive: bool = False) -> Dict[str, Any]:
    """Search records by PAN, Name, or Mobile across all product tables in PostgreSQL."""
    products = list_products_pg()
    all_results = []
    
    # Check if at least one search parameter is provided
    if not any([pan, name, mobile]):
        return {
            'success': False,
            'error': 'At least one search parameter (PAN, Name, or Mobile) is required',
            'total_records': 0,
            'records': []
        }

    with get_db_connection() as conn:
        for product in products:
            disbursed_table = get_table_name(product, 'disbursed')
            collection_table = get_table_name(product, 'collection')

            try:
                with conn.cursor() as cur:
                    # Build WHERE clause dynamically
                    where_conditions = []
                    params = []
                    
                    if pan:
                        where_conditions.append('LOWER(d."Pancard") = LOWER(%s)')
                        params.append(pan)
                    if name:
                        where_conditions.append('LOWER(d."Name") LIKE LOWER(%s)')
                        params.append(f'%{name}%')
                    if mobile:
                        where_conditions.append('LOWER(d."Mobile") = LOWER(%s)')
                        params.append(mobile)
                    
                    where_clause = ' OR '.join(where_conditions)
                    
                    query = f"""
                        SELECT d.*, c."Collected_Date", c."Collected_Amount"
                        FROM {disbursed_table} d
                        LEFT JOIN {collection_table} c 
                            ON d."LeadID" = c."LeadID" AND d."Loan_No" = c."Loan_No"
                        WHERE {where_clause}
                        ORDER BY d."Repay_Date"
                    """
                    
                    cur.execute(query, tuple(params))

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
        'name': name,
        'mobile': mobile,
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


# Authentication and Logging Functions

def create_users_table():
    """Create users table for authentication."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Users table created/verified")


def create_activity_logs_table():
    """Create activity logs table for tracking user actions."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    details TEXT,
                    ip_address VARCHAR(45),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Activity logs table created/verified")


def init_auth_tables():
    """Initialize authentication tables."""
    create_users_table()
    create_activity_logs_table()


def create_user(username: str, password: str, role: str = 'user') -> bool:
    """Create a new user with hashed password."""
    import hashlib
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO users (username, password_hash, role)
                    VALUES (%s, %s, %s)
                """, (username, password_hash, role))
                conn.commit()
                logger.info(f"User created: {username} (role: {role})")
                return True
            except psycopg2.IntegrityError:
                logger.warning(f"Username already exists: {username}")
                return False


def admin_create_user(username: str, password: str, role: str = 'user', created_by: str = None) -> Dict[str, Any]:
    """Admin creates a new user with generated password. Returns the generated password."""
    import hashlib
    import secrets
    import string
    
    # Generate random password if not provided
    if not password:
        password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO users (username, password_hash, role)
                    VALUES (%s, %s, %s)
                """, (username, password_hash, role))
                conn.commit()
                
                # Log admin activity
                log_activity(created_by or 'admin', "CREATE_USER", f"Created user: {username} (role: {role})")
                
                logger.info(f"Admin created user: {username} (role: {role})")
                return {
                    'success': True,
                    'username': username,
                    'password': password,
                    'role': role,
                    'message': f"User '{username}' created successfully with password: {password}"
                }
            except psycopg2.IntegrityError:
                logger.warning(f"Username already exists: {username}")
                return {
                    'success': False,
                    'error': f"Username '{username}' already exists"
                }


def get_all_users() -> List[Dict[str, Any]]:
    """Get all users (for admin)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, username, role, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cur.fetchall()]


def delete_user(username: str, deleted_by: str = None) -> bool:
    """Delete a user (admin only)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            conn.commit()
            
            if cur.rowcount > 0:
                log_activity(deleted_by or 'admin', "DELETE_USER", f"Deleted user: {username}")
                logger.info(f"User deleted: {username}")
                return True
            return False


def reset_user_password(username: str, new_password: str = None, reset_by: str = None) -> Dict[str, Any]:
    """Reset user password (admin only). Returns the new password."""
    import hashlib
    import secrets
    import string
    
    # Generate random password if not provided
    if not new_password:
        new_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET password_hash = %s
                WHERE username = %s
            """, (password_hash, username))
            conn.commit()
            
            if cur.rowcount > 0:
                log_activity(reset_by or 'admin', "RESET_PASSWORD", f"Reset password for: {username}")
                logger.info(f"Password reset for user: {username}")
                return {
                    'success': True,
                    'username': username,
                    'new_password': new_password,
                    'message': f"Password reset for '{username}'. New password: {new_password}"
                }
            return {
                'success': False,
                'error': f"User '{username}' not found"
            }


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials and return user info."""
    import hashlib
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, username, role, created_at
                FROM users
                WHERE username = %s AND password_hash = %s
            """, (username, password_hash))
            
            user = cur.fetchone()
            if user:
                # Update last login
                cur.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE username = %s
                """, (username,))
                conn.commit()
                return dict(user)
            return None


def log_activity(username: str, action: str, details: str = None, ip_address: str = None):
    """Log user activity."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO activity_logs (username, action, details, ip_address)
                VALUES (%s, %s, %s, %s)
            """, (username, action, details, ip_address))
            conn.commit()
            logger.info(f"Activity logged: {username} - {action}")


def get_user_logs(username: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get activity logs for a user or all users."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if username:
                cur.execute("""
                    SELECT * FROM activity_logs
                    WHERE username = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (username, limit))
            else:
                cur.execute("""
                    SELECT * FROM activity_logs
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
            
            return [dict(row) for row in cur.fetchall()]


# Initialize auth tables on module load
init_auth_tables()
