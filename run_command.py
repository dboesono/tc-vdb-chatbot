import subprocess

def main():
    try:
        # Run the command to delete the vector database
        subprocess.run(["python", "delete_vector_db.py", "email_db"], check=True)
        
        # Run the Streamlit app
        subprocess.run(["streamlit", "run", "app2.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
