// Customer 360 Insight - JavaScript Application

let currentResults = [];

// Icons SVG
const icons = {
    upload: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>`,
    check: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>`,
    alert: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
    search: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>`,
    database: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/></svg>`,
    empty: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    checkAuth();
    loadDatabases();
    setupEventListeners();
});

// Check authentication and load user info
async function checkAuth() {
    try {
        const response = await fetch('/api/check-auth');
        const data = await response.json();

        if (data.authenticated) {
            document.getElementById('username').textContent = data.username;
        } else {
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        window.location.href = '/login';
    }
}

// Logout function
async function logout() {
    try {
        const response = await fetch('/api/logout', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Logout failed:', error);
        alert('Logout failed. Please try again.');
    }
}

// Event Listeners
function setupEventListeners() {
    document.getElementById('disbursedFile').addEventListener('change', function () {
        updateFileLabel('disbursedLabel', this.files[0]);
    });

    document.getElementById('collectionFile').addEventListener('change', function () {
        updateFileLabel('collectionLabel', this.files[0]);
    });
}

// Update file label
function updateFileLabel(labelId, file) {
    const label = document.getElementById(labelId);
    if (file) {
        label.innerHTML = `${icons.check}<span class="file-name">${file.name}</span><span class="file-hint">Click to change</span>`;
        label.classList.add('has-file');
    } else {
        label.innerHTML = `${icons.upload}<span class="file-name">Choose file</span><span class="file-hint">Drop file here or click to browse</span>`;
        label.classList.remove('has-file');
    }
}

// Set button loading state
function setButtonLoading(btnId, textId, isLoading) {
    const btn = document.getElementById(btnId);
    const text = document.getElementById(textId);
    if (isLoading) {
        btn.disabled = true;
        text.innerHTML = '<span class="spinner"></span> Processing...';
    } else {
        btn.disabled = false;
        const btnText = btnId === 'processBtn' ? 'Process Files' :
            btnId === 'searchBtn' ? 'Search' : 'Run Query';
        text.textContent = btnText;
    }
}

// Load databases
async function loadDatabases() {
    try {
        const response = await fetch('/api/databases');
        const data = await response.json();
        const container = document.getElementById('databasesList');

        if (data.databases.length === 0) {
            container.innerHTML = '<span style="color: #94a3b8;">No databases found. Upload files to create one.</span>';
        } else {
            container.innerHTML = data.databases.map(db =>
                `<span class="database-tag product-${db.toLowerCase()}"><span class="db-dot"></span>${db}</span>`
            ).join('');
        }
    } catch (error) {
        document.getElementById('databasesList').innerHTML = '<span style="color: #ef4444;">Failed to load databases</span>';
    }
}

// Toggle Upload Section
function toggleUploadSection() {
    const content = document.getElementById('uploadContent');
    const icon = document.getElementById('uploadToggleIcon');

    content.classList.toggle('hidden');
    icon.classList.toggle('expanded');
}

// Process files
async function processFiles() {
    const disbursedFile = document.getElementById('disbursedFile').files[0];
    const collectionFile = document.getElementById('collectionFile').files[0];

    if (!disbursedFile || !collectionFile) {
        showAlert('uploadResult', 'Please select both files', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('disbursed', disbursedFile);
    formData.append('collection', collectionFile);

    setButtonLoading('processBtn', 'processBtnText', true);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            showAlert('uploadResult', data.message, 'success');
            loadDatabases();
        } else {
            showAlert('uploadResult', data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showAlert('uploadResult', 'Network error: ' + error.message, 'error');
    } finally {
        setButtonLoading('processBtn', 'processBtnText', false);
    }
}

// Search PAN
async function searchPAN() {
    const pan = document.getElementById('panInput').value.trim();

    if (!pan) {
        showAlert('searchResult', 'Please enter a PAN', 'error');
        return;
    }

    setButtonLoading('searchBtn', 'searchBtnText', true);

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pan: pan })
        });
        const data = await response.json();

        if (data.success) {
            currentResults = data.records;
            displayResults('searchResult', data.records, data.pan, data.total_records, false); // static columns
        } else {
            showAlert('searchResult', data.error || 'Search failed', 'error');
        }
    } catch (error) {
        showAlert('searchResult', 'Network error: ' + error.message, 'error');
    } finally {
        setButtonLoading('searchBtn', 'searchBtnText', false);
    }
}

// Run SQL query (SELECT only)
async function runQuery() {
    const query = document.getElementById('sqlInput').value.trim();

    if (!query) {
        showAlert('queryResult', 'Please enter a SQL query', 'error');
        return;
    }

    setButtonLoading('queryBtn', 'queryBtnText', true);

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await response.json();

        if (data.success) {
            currentResults = data.records;
            displayResults('queryResult', data.records, null, data.total_records, true); // dynamic columns
            document.getElementById('exportBtn').classList.remove('hidden');
        } else {
            showAlert('queryResult', data.error || 'Query failed', 'error');
            document.getElementById('exportBtn').classList.add('hidden');
        }
    } catch (error) {
        showAlert('queryResult', 'Network error: ' + error.message, 'error');
        document.getElementById('exportBtn').classList.add('hidden');
    } finally {
        setButtonLoading('queryBtn', 'queryBtnText', false);
    }
}

// Export results
async function exportResults() {
    if (currentResults.length === 0) {
        alert('No results to export');
        return;
    }

    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ records: currentResults })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'query_result.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('Export failed');
        }
    } catch (error) {
        alert('Export error: ' + error.message);
    }
}

// Display results in table with optional dynamic columns
function displayResults(elementId, records, pan, total, useDynamicColumns = false) {
    const container = document.getElementById(elementId);

    if (records.length === 0) {
        container.innerHTML = `<div class="empty-state">${icons.empty}<p>No records found</p></div>`;
        return;
    }

    // Static column configuration for PAN search
    const staticColumnConfig = [
        { key: 'Name', label: 'Name' },
        { key: 'Email', label: 'Email' },
        { key: 'Mobile', label: 'Mobile' },
        { key: 'LoanAmount', label: 'Loan Amount' },
        { key: 'RepayDate', label: 'Repay Date' },
        { key: 'CollectionDate', label: 'Collection Date' },
        { key: 'PaymentStatus', label: 'Payment Status' },
        { key: 'Status', label: 'Status' },
        { key: 'Product', label: 'Product' }
    ];

    let columnOrder;

    if (useDynamicColumns) {
        // Extract all unique columns from records for SQL queries
        const columnSet = new Set();
        columnOrder = [];

        records.forEach(record => {
            Object.keys(record).forEach(key => {
                const lowerKey = key.toLowerCase();
                if (!columnSet.has(lowerKey)) {
                    columnSet.add(lowerKey);
                    columnOrder.push(key);
                }
            });
        });
    } else {
        // Use static columns for PAN search
        columnOrder = staticColumnConfig.map(col => col.key);
    }

    // Define display names for common columns
    const displayNames = {
        'name': 'Name',
        'email': 'Email',
        'mobile': 'Mobile',
        'loanamount': 'Loan Amount',
        'loan_amount': 'Loan Amount',
        'repaydate': 'Repay Date',
        'repay_date': 'Repay Date',
        'collectiondate': 'Collection Date',
        'collected_date': 'Collection Date',
        'paymentstatus': 'Payment Status',
        'status': 'Status',
        'product': 'Product',
        'pancard': 'PAN',
        'leadid': 'Lead ID',
        'loan_no': 'Loan No',
        'branch': 'Branch',
        'disbursed_date': 'Disbursed Date',
        'collected_amount': 'Collected Amount',
        'created_at': 'Created At'
    };

    // Get display name for a column
    function getDisplayName(col) {
        if (!useDynamicColumns) {
            const staticCol = staticColumnConfig.find(c => c.key === col);
            if (staticCol) return staticCol.label;
        }
        return displayNames[col.toLowerCase()] || col;
    }

    // Build table HTML
    let html = `
        <div class="stats-bar">
            ${pan ? `<div class="stat-item"><span class="stat-label">PAN</span><span class="stat-value">${pan}</span></div>` : ''}
            <div class="stat-item"><span class="stat-label">Total Records</span><span class="stat-value">${total}</span></div>
        </div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>${columnOrder.map(col => `<th>${getDisplayName(col)}</th>`).join('')}</tr>
                </thead>
                <tbody>
                    ${records.map((record) => {
        const product = record.Product || record.product || '';
        const status = (record.PaymentStatus || record.paymentstatus || '').toLowerCase().replace(/ /g, '_');
        const rowClass = product ? `product-${product.toLowerCase()}` : '';

        return `<tr class="${rowClass}">${columnOrder.map(col => {
            // Case-insensitive key lookup
            const actualKey = Object.keys(record).find(k => k.toLowerCase() === col.toLowerCase()) || col;
            let value = record[actualKey];
            if (value === null || value === undefined || value === '') value = '-';

            // Special formatting for PaymentStatus
            const isPaymentStatus = col.toLowerCase() === 'paymentstatus';
            if (isPaymentStatus && value !== '-') {
                return `<td><span class="status-badge status-${status}">${value}</span></td>`;
            }

            return `<td>${value}</td>`;
        }).join('')}</tr>`;
    }).join('')}
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = html;
}

// Show alert
function showAlert(elementId, message, type) {
    const container = document.getElementById(elementId);
    container.innerHTML = `<div class="alert alert-${type}">${icons.alert} ${message}</div>`;
}
