import os
import sys
import logging
import tcvectordb
from tcvectordb.model.enum import ReadConsistency

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set your Tencent Cloud VectorDB configuration.
# Replace <your-vector-db-url> with your actual external URL.
TENCENT_VECTORDB_URL = "http://gz-vdb-dbtf4m82.sql.tencentcdb.com:8100"
TENCENT_VECTORDB_USERNAME = "root"
# Ensure your API key is set in your environment, or replace os.environ.get("TC_API_KEY")
API_KEY = 'Cp6MNT6erkUntajMfDKQk6DnHNhf40PiNVeKgobr'

# Create the VectorDB client.
client = tcvectordb.RPCVectorDBClient(
    url=TENCENT_VECTORDB_URL,
    username=TENCENT_VECTORDB_USERNAME,
    key=API_KEY,
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30
)

def delete_vector_database(database_name):
    """
    Deletes the specified vector database.
    """
    try:
        result = client.drop_database(database_name=database_name)
        logging.info(f"Database '{database_name}' deleted successfully.")
        print(f"Database '{database_name}' deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting database '{database_name}': {e}")
        print(f"Error deleting database '{database_name}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_vector_db.py <database_name>")
        sys.exit(1)
    
    database_name = sys.argv[1]
    delete_vector_database(database_name)
