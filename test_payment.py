import sqlite3
import os
from datetime import datetime

DB = os.path.join('reports','loan_applications.db')

if not os.path.exists(DB):
    print('DB not found at', DB)
    raise SystemExit(1)

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Ensure payments table exists (should be created by app init)
c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='payments' ''')
if not c.fetchone():
    c.execute('''
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER,
            timestamp TEXT,
            months_paid INTEGER,
            amount_paid REAL,
            payer TEXT
        )
    ''')
    conn.commit()

# Find a recent application
c.execute('SELECT id, amount, monthly_payment FROM loan_applications ORDER BY id DESC LIMIT 1')
row = c.fetchone()
if not row:
    # Insert a dummy application
    ts = datetime.now().isoformat()
    c.execute('''INSERT INTO loan_applications (timestamp, full_name, contact, amount, months, interest_rate, monthly_payment, verification) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (ts, 'Test User', '09171234567', 10000.0, 12, 5.0, 856.07, None))
    conn.commit()
    app_id = c.lastrowid
    monthly = 856.07
    print('Inserted dummy application id', app_id)
else:
    app_id = row['id']
    monthly = row['monthly_payment'] if row['monthly_payment'] is not None else row['amount']
    print('Using existing application id', app_id)

# Insert payment for 1 month
ts = datetime.now().isoformat()
c.execute('INSERT INTO payments (application_id, timestamp, months_paid, amount_paid, payer) VALUES (?, ?, ?, ?, ?)',
          (app_id, ts, 1, monthly, 'test-runner'))
conn.commit()
print('Inserted payment for application', app_id, 'amount', monthly)

# Show last 10 payments
print('\nLast payments:')
for r in c.execute('SELECT id, application_id, timestamp, months_paid, amount_paid, payer FROM payments ORDER BY id DESC LIMIT 10'):
    print(dict(r))

conn.close()
