#!/usr/bin/env python3
"""
Database reset script - recreates the database with the latest schema
"""
import os
import sqlite3

def reset_database():
    """Remove old database and let the API recreate it with new schema."""
    db_file = 'fingerprints.db'
    
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"✅ Removed old database: {db_file}")
    else:
        print(f"ℹ️  Database file not found: {db_file}")
    
    print("🔄 Database will be recreated when API starts")
    print("🚀 Please restart the API server to create new schema")

if __name__ == "__main__":
    reset_database()
