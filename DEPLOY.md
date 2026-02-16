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
       ⚠️ **Important:** The app module is `Dashboard`, not `app`. If you leave the default (`gunicorn app:app`), the deploy will fail with `ModuleNotFoundError: No module named 'app'`.
   - **Or** use **New + → Blueprint** and connect the repo; the repo’s `render.yaml` sets the start command for you.

4. **Wait for the first deploy**  
   Render will install dependencies and start the app. When the build succeeds, it will show a URL like `https://instacart-dashboard-xxxx.onrender.com`.

5. **Share that URL**  
   Anyone can open it. On the free tier the app may sleep after ~15 min of no use; the first visit after that might take 30–60 seconds to wake up.

### If you already created the service and got “No module named 'app'”

- In Render: open your service → **Settings** → **Build & Deploy**.
- Set **Start Command** to: `gunicorn Dashboard:server --bind 0.0.0.0:$PORT`
- Save and trigger a **Manual Deploy**.

### If your CSV is not in the repo

- Either add `final_data_instacart_400k.csv` to the repo and push, or  
- Use Render’s **Shell** (from the service page) to upload the file into the app’s working directory, then redeploy.

---

## Option 3: Deploy on GCP (Cloud Run)

Deploy from your machine with one command. You get a permanent URL on Google Cloud. Free tier is generous; you only pay if you exceed it.

### Prerequisites

1. **Google Cloud account** — https://cloud.google.com  
2. **Install Google Cloud SDK** — https://cloud.google.com/sdk/docs/install  
3. **Create a project** (or use an existing one):
   ```bash
   gcloud projects create YOUR_PROJECT_ID --name "Instacart Dashboard"
   gcloud config set project YOUR_PROJECT_ID
   ```
4. **Enable required APIs and log in:**
   ```bash
   gcloud auth login
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```

### Deploy from the project folder

From the repo root (where `Dashboard.py` and `requirements.txt` are):

```bash
cd "/Users/chinmaydhamapurkar/Instacart Dashboard/Instacart_Market-Analysis"

gcloud run deploy instacart-dashboard \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi
```

- **First time:** Cloud Build will build from source (uses your `requirements.txt` and **Procfile** to start the app). This can take a few minutes.  
- You’ll get a URL like `https://instacart-dashboard-xxxxx-uc.a.run.app`. Share that link.

**Notes:**

- The **Procfile** in this repo tells Cloud Run to run:  
  `gunicorn Dashboard:server --bind 0.0.0.0:$PORT`  
  Cloud Run sets `$PORT` (usually 8080).
- Include `final_data_instacart_400k.csv` in the folder (or in the repo) so the app can find it when it runs.
- To redeploy after code changes, run the same `gcloud run deploy` command again.

### Optional: set project and region

```bash
gcloud config set project YOUR_PROJECT_ID
# Use a region near you, e.g. us-east1, europe-west1
gcloud run deploy instacart-dashboard --source . --region us-east1 --allow-unauthenticated --memory 1Gi
```

---

## Quick reference

| Goal              | Use           | Time   |
|-------------------|---------------|--------|
| Share in 2 min    | ngrok         | ~2 min |
| Always-on link    | Render        | ~10 min first time |
| Deploy from CLI   | GCP Cloud Run | ~5 min after setup |

**Start command for production (Procfile):**  
`gunicorn Dashboard:server --bind 0.0.0.0:$PORT`  
(Locally you can run `python3 Dashboard.py`.)
