import psycopg2

import legal_workflow_generator.config.values as config

print("connecting...")
conn = psycopg2.connect(
    dbname="postgres",
    user=config.DB_USER,
    password=config.PGPASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT,
    connect_timeout=5,
)
print("connected!")
conn.close()
print("done")
