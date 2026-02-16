# Thin wrapper so Render's default start command (gunicorn app:app) works.
# The real app is in Dashboard.py; this just exposes its Flask server as "app".
from Dashboard import server

app = server
