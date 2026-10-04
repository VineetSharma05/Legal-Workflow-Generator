import psycopg2
import legal_workflow_generator.config.values as c

conn = psycopg2.connect(dbname=c.DB_NAME, user=c.DB_USER, password=c.PGPASSWORD,
                        host=c.DB_HOST, port=c.DB_PORT)
cur = conn.cursor()

cur.execute("select count(*), count(embedding) from laws")
total, embedded = cur.fetchone()
print(f"rows: {total}, with embeddings: {embedded}")

cur.execute("select split_part(provision_id, '_sec', 1) p, domain, count(*) from laws group by 1, 2 order by 3 desc")
for row in cur.fetchall():
    print(row)
