import psycopg2
import os

if "PGPASSWORD" not in os.environ:
    print("PGPASSWORD env not set. Please set it and then run the script")
    exit(1)

print("connecting...")
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password=os.environ["PGPASSWORD"],
    host="localhost",
    port=5432,
    connect_timeout=5,
)
print("connected!")
conn.close()
print("done")
