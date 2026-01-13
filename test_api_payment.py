import sqlite3
import os
import json
import urllib.request
import urllib.parse
import http.cookiejar

DB = os.path.join('reports','loan_applications.db')
if not os.path.exists(DB):
    print('DB not found at', DB)
    raise SystemExit(1)

# find latest application id and monthly_payment
conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute('SELECT id, monthly_payment, amount FROM loan_applications ORDER BY id DESC LIMIT 1')
row = c.fetchone()
conn.close()
if not row:
    print('No applications found in DB')
    raise SystemExit(1)
app_id, monthly_payment, amount = row
monthly_payment = float(monthly_payment) if monthly_payment is not None else float(amount) if amount is not None else 0.0
print('Latest application id:', app_id, 'monthly:', monthly_payment)

BASE = 'http://127.0.0.1:5000'

# Setup cookie-capable opener
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

def post_json(path, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=10) as resp:
        return resp.read().decode('utf-8'), resp.getcode()

def get(path):
    req = urllib.request.Request(BASE + path)
    with opener.open(req, timeout=10) as resp:
        return resp.read().decode('utf-8'), resp.getcode()

# login
try:
    body, code = post_json('/login', {'username': 'admin', 'password': 'jethro123'})
    print('Login status:', code)
    try:
        print('Login response:', json.loads(body))
    except:
        print('Login raw:', body[:200])
except Exception as e:
    print('Login failed:', e)
    raise

# post payment
pay_payload = {'months': 1, 'amount': monthly_payment}
try:
    body2, code2 = post_json(f'/pay-application/{app_id}', pay_payload)
    print('Post payment status:', code2)
    try:
        print('Post payment response:', json.loads(body2))
    except:
        print('Post raw:', body2[:200])
except Exception as e:
    print('Post payment failed:', e)
    raise

# fetch payments
try:
    body3, code3 = get(f'/payments/{app_id}')
    print('Get payments status:', code3)
    try:
        print('Payments:', json.loads(body3))
    except:
        print('Payments raw:', body3[:500])
except Exception as e:
    print('Get payments failed:', e)
    raise
