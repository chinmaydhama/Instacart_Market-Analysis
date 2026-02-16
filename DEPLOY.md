# Deploy & share your dashboard

Two options: **instant share** (temporary URL) or **permanent deploy** (always-on link).

---

## Option 1: Instant share (2 minutes) — ngrok

Best when you want to share **right now** and your app is running on your machine.

1. **Run the dashboard locally**
   ```bash
   cd "/Users/chinmaydhamapurkar/Instacart Dashboard/Instacart_Market-Analysis"
   python3 Dashboard.py
   ```

2. **Install ngrok** (one-time)  
   - Download: https://ngrok.com/download  
   - Or: `brew install ngrok` (Mac)

3. **In a new terminal, start the tunnel**
   ```bash
   ngrok http 8050
   ```

4. **Share the URL**  
   ngrok will show something like `https://abc123.ngrok.io` — send that link. Anyone can open it while your app and ngrok are running.

**Note:** The link stops working when you close the terminal or stop ngrok. Your CSV must be present locally.

---

## Option 2: Permanent deploy (free) — Render

Gives you a **permanent URL** (e.g. `https://your-app.onrender.com`) that stays up 24/7 on Render’s free tier.

### One-time setup

1. **Put your project on GitHub**
   - Create a repo and push this folder (including `final_data_instacart_400k.csv` and `requirements.txt`).
   - If the CSV is large, you can use Git LFS or add it from the Render shell later; for a quick share, committing it is simplest.

2. **Sign up at Render**
   - Go to https://render.com and sign up (free; GitHub login is fine).

3. **Create a new Web Service**
   - Dashboard → **New +** → **Web Service**.
   - Connect your GitHub account and select the repo (and the branch that has the app).
   - Use these settings:
     - **Name:** e.g. `instacart-dashboard`
     - **Runtime:** Python 3
     - **Build command:** `pip install -r requirements.txt`
     - **Start command:** `gunicorn Dashboard:server --bind 0.0.0.0:$PORT`
   - Click **Create Web Service**.

4. **Wait for the first deploy**  
   Render will install dependencies and start the app. When the build succeeds, it will show a URL like `https://instacart-dashboard-xxxx.onrender.com`.

5. **Share that URL**  
   Anyone can open it. On the free tier the app may sleep after ~15 min of no use; the first visit after that might take 30–60 seconds to wake up.

### If your CSV is not in the repo

- Either add `final_data_instacart_400k.csv` to the repo and push, or  
- Use Render’s **Shell** (from the service page) to upload the file into the app’s working directory, then redeploy.

---

## Quick reference

| Goal              | Use        | Time   |
|-------------------|------------|--------|
| Share in 2 min    | ngrok      | ~2 min |
| Always-on link    | Render     | ~10 min first time |

**Start command for production (already set in the app):**  
`gunicorn Dashboard:server --bind 0.0.0.0:$PORT`  
(Render sets `$PORT`; locally you can run `python3 Dashboard.py`.)
