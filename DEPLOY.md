# Deploying RB Model — full steps (free, ~15 minutes)

Two free accounts, no credit card, no server to maintain. Once deployed you
get a public URL — you or anyone you give it to enters a team ID and clicks
Run Model.

## What this costs: confirmed, item by item

| Piece | Cost | Basis |
|---|---|---|
| GitHub repository (public) | Free | GitHub's standard free public-repo tier |
| Streamlit Community Cloud hosting | Free | Streamlit's own cloud page states Community Cloud is "Totally free" for public apps, no credit card requested at signup |
| Official FPL API (`fantasy.premierleague.com/api`) | Free | Public, no API key or account needed for the bootstrap/fixtures/entry/entry-picks/entry-history endpoints this app uses. (The API has no official published ToS — see the caveat below.) |
| Google Fonts (Fraunces / IBM Plex) | Free | Google Fonts is a free, ad-free web font service |
| PuLP + CBC solver (squad optimizer) | Free | Open-source, MIT-license-compatible, bundled with the `pulp` PyPI package |

**One honest caveat, not a blocker:** the official FPL API has no
published, official terms-of-service document — every FPL community tool
(including the well-known ones) runs on the same undocumented-but-public
basis. This has been true for years and is how the entire FPL tooling
ecosystem operates. It is not a paid/licensed API and nothing about this
app asks for elevated or authenticated access — it only reads the same
public endpoints your browser reads when you open fantasy.premierleague.com.

**Public-app privacy note:** Streamlit Community Cloud's free tier is
public apps only — anyone with the URL can open it and enter any team ID
(not just yours), the same as you'd get with any link you hand out. That
was the explicit design decision earlier ("someone is having the URL and
add the team ID then start running"). If you later want a private/login
version, that's a paid-tier change, not something this free deploy does.

---

## Step 1 — Create a free GitHub account (skip if you have one)

1. Go to [github.com/signup](https://github.com/signup) and create an
   account. Free.

## Step 2 — Create the repository

1. Click the **+** in GitHub's top-right corner → **New repository**.
2. Name it `rb-model` (or anything you like).
3. Set it to **Public** (Streamlit Community Cloud's free tier requires this).
4. Leave "Add a README" unchecked (you already have one). Click **Create repository**.

## Step 3 — Upload the files

**Easiest path — no git needed:**

1. On the new repo's page, click **uploading an existing file**.
2. Drag in every file from this delivered folder — keep the `.streamlit`
   folder structure intact (GitHub's uploader preserves folder paths if you
   drag the whole folder, or create the `.streamlit` folder manually and
   upload `config.toml` into it if your browser flattens it).
3. Scroll down, write a commit message ("Initial RB Model upload"), click
   **Commit changes**.

**If you're comfortable with git instead:**

```bash
cd rb_model_app
git init
git add .
git commit -m "Initial RB Model upload"
git branch -M main
git remote add origin https://github.com/<your-username>/rb-model.git
git push -u origin main
```

Either way, confirm on GitHub afterward that `.streamlit/config.toml`
actually landed inside a `.streamlit` folder, not as a loose file — a flattened
upload is the one common mistake here and it silently breaks the theme.

## Step 4 — Create a free Streamlit Community Cloud account

1. Go to [share.streamlit.io](https://share.streamlit.io) (or
   [streamlit.io/cloud](https://streamlit.io/cloud) → "Sign up").
2. Sign up/sign in **with your GitHub account** — this is also how it gets
   permission to read your repo. Free, no card requested.

## Step 5 — Deploy

1. Click **New app** (top right).
2. Repository: select `<your-username>/rb-model`.
3. Branch: `main`.
4. Main file path: `app.py`.
5. Click **Deploy**.
6. Wait 2–5 minutes on first deploy — it's installing `requirements.txt`
   into a fresh container. You'll see a build log; a successful one ends
   with the app rendering the RB Model gate screen.

You now have a URL like `https://rb-model-<random>.streamlit.app`. That's
the link — bookmark it, share it, whatever you like.

## Step 6 — Test it

1. Open the URL.
2. Enter FPL ID `26073` (2026/27 season) at the gate screen, click **Unlock**.
3. In the sidebar: leave Style on "Calculated Maverick", Hit stance on
   "Hit if worth it", click **Run Model**.
4. You should see the verdict card, chip rack, pitch view, transfer table,
   captaincy pick, and season ledger populate with live data within a few
   seconds.

If step 6 fails, see Troubleshooting below before assuming something's
architecturally wrong — the most common first-deploy issues are quick fixes.

## Updating the model later

Edit `model_config.yaml` (a weight/threshold) or any `.py` file (new logic)
directly in GitHub's web editor (pencil icon on any file) or push from your
machine — either way, Streamlit Community Cloud watches the repo and
auto-redeploys on every push, typically live within under a minute. (It
rate-limits to five app updates per minute if you push a burst of commits —
irrelevant at normal single-edit-at-a-time usage.)

## Sleeping / waking

Free-tier Community Cloud apps that go a while with no visits enter a
sleeping state; the next visitor sees a "This app has gone to sleep, wake
it up" button and a roughly 30–60 second wait while it restarts. No cost,
no data lost — `model_config.yaml`, your code, everything's still exactly
as you left it. It just isn't running a live container 24/7 when nobody's
looking, which is part of how the free tier stays free.

## Troubleshooting

- **"Couldn't reach the official FPL API"** on the gate screen or Run —
  the API is free and normally very reliable, but if the official site
  itself is down (rare, usually deadline-day load spikes) this app will be
  down too, since it has no cached fallback for entry-specific data. Retry
  in a few minutes.
- **Team ID not found** — double-check the number. It's the digits in the
  URL when you open **Points** on your team on the official site
  (`fantasy.premierleague.com/entry/<this number>/event/...`).
- **PuLP/CBC solver error in the logs** — very rare on Community Cloud
  (CBC ships inside the `pulp` package, no separate system install needed),
  but if it happens, the Team Rating % section will show "Not computed this
  run" rather than crashing the whole app — everything else still works.
- **Player photos not loading** — they hotlink the official
  `resources.premierleague.com` image CDN; a broken image quietly falls
  back to an initials avatar (built into the page, no fix needed on your
  end). If *every* photo is falling back, the CDN path likely changed —
  check the URL pattern in `app.py`'s `_photo_url()` against a photo URL
  from the live site and update if needed.
- **Redeploy didn't pick up your edit** — check you pushed/committed to the
  `main` branch specifically (or whichever branch you selected in Step 5).
