# Homicide_Agent

Lightweight supervisor-router multi-agent pipeline for targeted web data extraction and general web Q&A.

## Summary

Homicide_Agent is a Python project that implements a supervisor-style router to direct queries to either:
- a generalist agent (for general web Q&A), or
- a specialist agent (focused on extracting homicide/statistics-style data from official sources for specific cities).

The code demonstrates a pragmatic scraping + LLM-driven extraction pipeline:
- web search (Tavily),
- page/download fetch (requests, Selenium, pandas),
- PDF handling (PyMuPDF / Fitz + OCR fallback),
- HTML cleaning (BeautifulSoup),
- LLM-based structured-extraction into JSON (Google Generative API via langchain_google_genai).

This repo is experimental / research-focused and optimized for lower-cost LLM usage (`gemma-3-27b-it` in the code).

## Key files

- main.py — the active entry and core logic:
  - creates a generalist agent using hub-pulled prompt and TavilySearchResults
  - builds a specialist agent with an explicit state machine (`SpecialistState`) and nodes:
    - search_node: queries web for authoritative sources for an objective + city
    - select_urls_node: ranks/filters candidate urls
    - scrape_node: fetches CSV/XLSX via pandas, PDFs via requests, or renders pages with headless Chrome (Selenium)
    - distill_content_node: extracts main text from HTML and runs OCR on PDFs when necessary (PyMuPDF -> image -> pytesseract)
    - analyze_node: asks the specialist LLM to return only JSON matching a data schema
    - pry_deeper_node: finds deeper links on pages when a promising "data scent" is detected
  - state objects record `findings`, `urls_to_try`, `tried_urls`, `raw_content`, `cleaned_content`, and `final_report`

## How it works (high level)

1. Supervisor receives user input (not fully implemented in the excerpt).
2. Supervisor decides whether to route to the generalist or specialist agent.
3. If specialist:
   - Specialist sets an `objective` and iterates over `cities` (e.g., New York, Los Angeles, New Orleans).
   - For each city, it runs the node pipeline: search → select urls → scrape → distill → analyze → (optionally) pry deeper and repeat until data is collected or attempts exhausted.
   - The LLM is used to convert cleaned text into structured JSON according to `data_schema`.
4. If generalist:
   - Generalist runs a REACT-style agent (via LangChain) with a web-search tool and returns conversational answers.

## Requirements

At a minimum the code expects:
- Python 3.8+
- pip installable packages (examples):
  - langchain, langchain_core, langchain_google_genai, langchain_community, langgraph
  - requests, beautifulsoup4, pandas, selenium, python-dotenv
  - PyMuPDF (fitz), pillow, pytesseract
  - Tavily search tool dependency (langchain_community.tools.tavily_search)
- A ChromeDriver compatible with the installed Chrome version for Selenium headless rendering
- Google Generative API credentials (for ChatGoogleGenerativeAI)
- Optional: Tesseract OCR binary installed on the machine (for pytesseract)

Sample install:
```bash
python -m pip install -r requirements.txt
# or install individually:
pip install langchain langchain_google_genai langchain_community requests beautifulsoup4 pandas selenium python-dotenv pymupdf pillow pytesseract
```

Note: The code uses `gemma-3-27b-it` via ChatGoogleGenerativeAI — ensure the environment variables / credentials required by your Google GenAI client are configured (see below).

## Environment / Secrets

Place keys and configuration in a `.env` file and load them via python-dotenv. Typical entries:
- GOOGLE_API_KEY or whatever the langchain_google_genai client requires
- Any other API keys required by third‑party search tools

## Running

The repository does not include a single fully wired CLI in the visible snippet. Typical workflow:
1. Export environment variables or create `.env` with required keys.
2. Ensure ChromeDriver is installed and on PATH (for Selenium rendering).
3. Run main:
```bash
python main.py
```
4. Interact via whatever CLI/entrypoint the repo wires (or extend main.py to prompt for queries).

## Developer notes / caveats

- Several helper imports and variables are referenced in the code excerpt (e.g., `io`, `pytesseract`, `json`) — ensure they are imported at top of file where needed.
- The code uses headless Chrome for dynamic sites. For robust operation consider:
  - setting ChromeDriver path,
  - increasing timeouts,
  - adding retries and backoff.
- PDF handling currently renders pages to images and runs OCR. If PDFs include extractable text, consider using PyMuPDF text extraction before OCR for better accuracy and speed.
- The HTML distillation chooses the largest content block but that logic has placeholder/incomplete lines — review and harden selection heuristics to avoid noise.
- The LLM extraction step expects strict JSON output. Add sanitization or fallback parsing to avoid crashes on model anomalies.
- Rate limits, CAPTCHAs, and robots.txt: this code does not handle all legal/ethical constraints. Respect target site policies.

