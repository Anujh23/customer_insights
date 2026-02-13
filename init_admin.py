"""
Initialize default admin user for Customer 360 Insight.
Run this script once to create the admin user.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from database import create_user


def init_admin():
    """Create default admin user."""
    # Default admin credentials - CHANGE THESE AFTER FIRST LOGIN!
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    
    print(f"Creating admin user: {admin_username}")
    
    success = create_user(admin_username, admin_password, role="admin")
    
    if success:
        print(f"✅ Admin user '{admin_username}' created successfully!")
        print("⚠️  IMPORTANT: Change the default password after first login!")
    else:
        print(f"⚠️  Admin user '{admin_username}' already exists or error occurred.")
        print("   If you need to reset the password, delete the user from the database first.")


if __name__ == "__main__":
    init_admin()
