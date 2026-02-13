"""
Customer 360 Insight - Flask Web Application
PostgreSQL backend for Render deployment
"""
import logging
import os
import tempfile
from pathlib import Path
from io import StringIO

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

from database import (
    process_uploaded_files_pg,
    search_pan_pg,
    list_products_pg,
    run_sql_query_pg,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max file size

# Temporary directory for file exports
TEMP_DIR = Path(tempfile.gettempdir()) / "data_harbour"
TEMP_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Upload disbursed and collection CSVs, process into PostgreSQL."""
    if "disbursed" not in request.files or "collection" not in request.files:
        return jsonify({"error": "Both disbursed and collection files required"}), 400

    disbursed_file = request.files["disbursed"]
    collection_file = request.files["collection"]

    if disbursed_file.filename == "" or collection_file.filename == "":
        return jsonify({"error": "No files selected"}), 400

    try:
        disbursed_df = pd.read_csv(disbursed_file, dtype=str)
        collection_df = pd.read_csv(collection_file, dtype=str)

        # Infer product from loan number prefix (e.g., ELI212... -> ELI)
        loan_no = str(disbursed_df.iloc[0].get('Loan No', '')).strip()
        product = loan_no[:3].upper() if len(loan_no) >= 3 else 'UNK'

        result = process_uploaded_files_pg(disbursed_df, collection_df, product)
        logger.info(f"Uploaded {product}: {result['disbursed_count']} disbursed, {result['collection_count']} collection")

        return jsonify({
            "success": True,
            "product": result['product'],
            "message": f"Product DB ready: {result['product']} ({result['disbursed_count']} disbursed, {result['collection_count']} collection records)"
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search_pan():
    """Search PAN across all product databases."""
    data = request.get_json()
    pan = data.get("pan", "").strip()

    if not pan:
        return jsonify({"error": "PAN cannot be empty"}), 400

    try:
        result = search_pan_pg(pan)
        logger.info(f"PAN search: {pan} - {result['total_records']} records found")
        return jsonify(result)

    except Exception as e:
        logger.error(f"PAN search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export_results():
    """Export query results to CSV."""
    data = request.get_json()
    records = data.get("records", [])

    if not records:
        return jsonify({"error": "No records to export"}), 400

    try:
        df = pd.DataFrame(records)
        output_path = TEMP_DIR / "export_result.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(records)} records to CSV")

        return send_file(
            output_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="query_result.csv"
        )

    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/databases", methods=["GET"])
def list_databases():
    """List all products in database."""
    try:
        products = list_products_pg()
        return jsonify({"databases": products})
    except Exception as e:
        logger.error(f"Database list error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Get port from environment variable for Render deployment
    port = int(os.environ.get("PORT", 5000))
    # Don't use debug mode in production
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
