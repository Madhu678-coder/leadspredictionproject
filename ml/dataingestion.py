import pandas as pd
from sqlalchemy import create_engine

def data_ingestion(csv_file_path, table_name='lead'):
    """
    Uploads data from a CSV file to a PostgreSQL table and retrieves the data back into a DataFrame.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        table_name (str): Name of the PostgreSQL table.

    Returns:
        pd.DataFrame: DataFrame containing data retrieved from the PostgreSQL table.
    """
    try:
        # Step 1: Create SQLAlchemy Engine
        engine = create_engine("postgresql+psycopg2://postgres:Madhu14777@localhost:5432/mydb5")

        # Step 2: Load CSV into DataFrame
        df = pd.read_csv(csv_file_path)

        # Step 3: Upload DataFrame to PostgreSQL Table
        df.to_sql(table_name, con=engine, index=False, if_exists='replace')
        print(f"✅ Data uploaded to PostgreSQL table '{table_name}' successfully.")

        # Step 4: Retrieve Data back from PostgreSQL into DataFrame
        query = f"SELECT * FROM {table_name}"
        df_from_db = pd.read_sql_query(query, con=engine)
        print(f"✅ Data retrieved from PostgreSQL table '{table_name}':")
        print(df_from_db.head())

        return df_from_db

    except Exception as e:
        print("❌ Error:", e)
        return None
