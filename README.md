# AI News Intelligence Agent — LangGraph Version

Same beautiful UI, now powered by a REAL LangGraph multi-step agent.

---

## Setup (3 steps)

### Step 1 — Install Python dependencies
Open terminal in this folder and run:
```
pip install -r requirements.txt
```

### Step 2 — Start the Python agent server
```
python agent.py
```
You should see:
```
==================================================
  LangGraph AI News Agent
  Server running at http://localhost:5000
  Open index.html in your browser
==================================================
```

### Step 3 — Open the frontend
- Open `index.html` in VS Code
- Right click → Open with Live Server
- OR just double-click `index.html` to open in browser
- Enter your Serper + Gemini API keys
- Type a query and click Run Agent!

---

## How the LangGraph Agent Works

```
User Query
    ↓
NODE 1: plan_searches
  → Agent uses Gemini to plan 3 targeted search queries
    ↓
NODE 2: execute_searches  
  → Runs all 3 queries on Serper API
  → Collects unique articles
    ↓
NODE 3: evaluate_results
  → Agent decides: is this enough info?
    ↓
[CONDITIONAL EDGE]
  → If NOT enough → NODE 4: search_more (extra search)
  → If enough     → NODE 5: generate_report
    ↓
NODE 5: generate_report
  → Gemini reads all articles
  → Writes full intelligence report
    ↓
DONE → Report sent to frontend
```

---

## API Keys

- Serper API: https://serper.dev (free tier: 2500 searches)
- Gemini API: https://aistudio.google.com (free tier available)

---

## Files

| File | Purpose |
|------|---------|
| agent.py | LangGraph agent + Flask API server |
| index.html | Beautiful frontend UI |
| requirements.txt | Python dependencies |
