import psycopg2
print('connecting...')
conn = psycopg2.connect(dbname='postgres', user='postgres', password='sanj2005', host='localhost', port=5432, connect_timeout=5)
print('connected!')
conn.close()
print('done')
