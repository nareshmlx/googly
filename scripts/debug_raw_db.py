
import psycopg
import settings
import time

print("--- DEBUG RAW PSYCOPG ---")
print(f"URL: {settings.DATABASE_URL}")

try:
    print("Connecting...")
    start = time.time()
    # Parse the URL manually to remove 'postgresql+psycopg://' prefix if present for psycopg3
    # Actually psycopg.connect accepts the URI directly usually, but let's check.
    url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    
    conn = psycopg.connect(url)
    print("Connected!")
    
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(f"Result: {cur.fetchone()}")
        
    conn.close()
    print("Connection closed.")
    print(f"Time: {time.time() - start:.2f}s")
    
except Exception as e:
    print(f"FAILED: {e}")
