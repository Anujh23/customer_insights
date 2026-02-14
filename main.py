"""
Customer 360 Insight - FastAPI Application
PostgreSQL backend with authentication and activity logging
"""
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException, Depends, status, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from jose import JWTError, jwt

from database import (
    process_uploaded_files_pg,
    search_pan_pg,
    list_products_pg,
    run_sql_query_pg,
    get_table_columns,
    verify_user,
    log_activity,
    get_user_logs,
    admin_create_user,
    get_all_users,
    delete_user,
    reset_user_password,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Customer 360 Insight", version="2.0.0")

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Temporary directory for file exports
TEMP_DIR = Path(tempfile.gettempdir()) / "data_harbour"
TEMP_DIR.mkdir(exist_ok=True)

# JWT Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: Optional[str] = None  # Auto-generated if not provided
    role: str = "user"


class ResetPasswordRequest(BaseModel):
    username: str
    new_password: Optional[str] = None  # Auto-generated if not provided


class SQLQueryRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    pan: Optional[str] = None
    name: Optional[str] = None
    mobile: Optional[str] = None


class ExportRequest(BaseModel):
    records: List[dict]


# JWT Token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        if username is None:
            return None
        return {"username": username, "role": role}
    except JWTError:
        return None


# Dependency to get current user from token
async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_data = verify_token(token)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_data


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    return request.client.host if request.client else "unknown"


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    token = request.cookies.get("access_token")
    if not token or not verify_token(token):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/api/login")
async def login(request: Request, login_data: LoginRequest):
    """Authenticate user and return token."""
    user = verify_user(login_data.username, login_data.password)
    
    if not user:
        log_activity(login_data.username, "LOGIN_FAILED", "Failed login attempt", get_client_ip(request))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    # Log successful login
    log_activity(user["username"], "LOGIN", "User logged in", get_client_ip(request))
    logger.info(f"User logged in: {user['username']}")
    
    response = JSONResponse({
        "success": True,
        "username": user["username"],
        "role": user["role"]
    })
    
    # Set cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        samesite="lax"
    )
    
    return response


@app.post("/api/logout")
async def logout(request: Request, current_user: dict = Depends(get_current_user)):
    """Logout user."""
    log_activity(current_user["username"], "LOGOUT", "User logged out", get_client_ip(request))
    logger.info(f"User logged out: {current_user['username']}")
    
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
    response.delete_cookie("access_token")
    return response


@app.get("/api/check-auth")
async def check_auth(request: Request):
    """Check if user is authenticated."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    user_data = verify_token(token)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    return {
        "authenticated": True,
        "username": user_data["username"],
        "role": user_data["role"]
    }


# Admin User Management
@app.post("/api/admin/users")
async def create_user(
    request: Request,
    user_data: CreateUserRequest,
    current_user: dict = Depends(get_admin_user)
):
    """Admin creates a new user."""
    result = admin_create_user(
        user_data.username,
        user_data.password,
        user_data.role,
        created_by=current_user["username"]
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.get("/api/admin/users")
async def list_users(current_user: dict = Depends(get_admin_user)):
    """Admin lists all users."""
    users = get_all_users()
    return {"success": True, "users": users}


@app.delete("/api/admin/users/{username}")
async def delete_user_endpoint(
    username: str,
    current_user: dict = Depends(get_admin_user)
):
    """Admin deletes a user."""
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    success = delete_user(username, deleted_by=current_user["username"])
    if not success:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    
    return {"success": True, "message": f"User '{username}' deleted"}


@app.post("/api/admin/users/{username}/reset-password")
async def reset_password(
    username: str,
    request: Request,
    reset_data: ResetPasswordRequest,
    current_user: dict = Depends(get_admin_user)
):
    """Admin resets user password."""
    result = reset_user_password(
        username,
        reset_data.new_password,
        reset_by=current_user["username"]
    )
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# Data APIs
@app.post("/api/upload")
async def upload_files(
    request: Request,
    disbursed: UploadFile = File(...),
    collection: UploadFile = File(...),
    current_user: dict = Depends(get_admin_user)
):
    """Upload disbursed and collection CSVs."""
    try:
        # Read CSV files
        disbursed_df = pd.read_csv(disbursed.file, dtype=str)
        collection_df = pd.read_csv(collection.file, dtype=str)
        
        # Infer product from loan number
        loan_no = str(
            disbursed_df.iloc[0].get('Loan No', disbursed_df.iloc[0].get('Loan_No', ''))
        ).strip()
        product = loan_no[:3].upper() if len(loan_no) >= 3 else 'UNK'
        
        result = process_uploaded_files_pg(disbursed_df, collection_df, product)
        
        log_activity(
            current_user["username"],
            "UPLOAD",
            f"Uploaded {product}: {result['disbursed_inserted']} disbursed, {result['collection_inserted']} collection",
            get_client_ip(request)
        )
        
        logger.info(f"Uploaded {product}: {result}")
        
        return {
            "success": True,
            "product": result['product'],
            "message": (
                f"Product DB ready: {result['product']} ("
                f"{result['disbursed_inserted']} inserted/{result['disbursed_updated']} updated disbursed, "
                f"{result['collection_inserted']} inserted/{result['collection_updated']} updated collection)"
            )
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def run_query(
    request: Request,
    query_data: SQLQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Run SQL query (SELECT only)."""
    query = query_data.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = run_sql_query_pg(query)
        
        log_activity(
            current_user["username"],
            "SQL_QUERY",
            f"Query: {query[:100]}",
            get_client_ip(request)
        )
        
        logger.info(f"Query executed: {query[:50]}... - {result['total_records']} records")
        return result
    except ValueError as e:
        logger.warning(f"Query validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_records(
    request: Request,
    search_data: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """Search records by PAN, Name, or Mobile across all products."""
    # Check at least one search parameter provided
    if not any([search_data.pan, search_data.name, search_data.mobile]):
        raise HTTPException(status_code=400, detail="At least one search parameter (PAN, Name, or Mobile) is required")
    
    try:
        result = search_pan_pg(
            pan=search_data.pan,
            name=search_data.name,
            mobile=search_data.mobile
        )
        
        log_activity(
            current_user["username"],
            "SEARCH",
            f"PAN: {search_data.pan}, Name: {search_data.name}, Mobile: {search_data.mobile}, Records: {result['total_records']}",
            get_client_ip(request)
        )
        
        logger.info(f"Search completed - {result['total_records']} records found")
        return result
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export")
async def export_results(
    export_data: ExportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Export results to CSV."""
    records = export_data.records
    
    if not records:
        raise HTTPException(status_code=400, detail="No records to export")
    
    try:
        df = pd.DataFrame(records)
        output_path = TEMP_DIR / "export_result.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(records)} records to CSV")
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="query_result.csv"
        )
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/databases")
async def list_databases(current_user: dict = Depends(get_current_user)):
    """List all products."""
    try:
        products = list_products_pg()
        return {"databases": products}
    except Exception as e:
        logger.error(f"Database list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/columns/{table_name}")
async def get_columns(table_name: str, current_user: dict = Depends(get_current_user)):
    """Get columns for a table."""
    try:
        columns = get_table_columns(table_name)
        return {
            "table": table_name,
            "columns": columns,
            "total_columns": len(columns)
        }
    except Exception as e:
        logger.error(f"Columns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User Activity Logs
@app.get("/api/user/logs")
async def get_logs(
    username: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get activity logs (admin sees all, users see own)."""
    target_user = username if current_user["role"] == "admin" else current_user["username"]
    
    try:
        logs = get_user_logs(target_user, limit)
        return {"success": True, "logs": logs, "total": len(logs)}
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import RedirectResponse

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
