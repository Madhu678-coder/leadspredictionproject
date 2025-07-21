from sqlalchemy import create_engine, text

# 🔧 Update with your DB credentials
DB_USER = "postgres"
DB_PASSWORD = "Madhu14777"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydb5"

# 🔗 Create engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ✅ SQL commands to add columns if not already there
alter_statements = [
    'ALTER TABLE lead_scoring ADD COLUMN IF NOT EXISTS "Prediction" TEXT;',
    'ALTER TABLE lead_scoring ADD COLUMN IF NOT EXISTS "Probability" DOUBLE PRECISION;',
    'ALTER TABLE lead_scoring ADD COLUMN IF NOT EXISTS "Confidence" TEXT;'
]

with engine.connect() as conn:
    for stmt in alter_statements:
        try:
            conn.execute(text(stmt))
            print(f"✅ Executed: {stmt}")
        except Exception as e:
            print(f"⚠️ Skipped or failed: {stmt} -> {e}")

print("✅ Table schema updated successfully!")
