# Ledger / Decode frontend

Premium React/Vite dashboard for the Women Tech Quest utility bill decoder. All frontend files live in this directory.

## Run locally

```powershell
cd E:\Women-Tech-Quest\frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## Backend contract

The current repository exposes the Python decoder as a CLI, not an HTTP server. The frontend therefore uses an explicit adapter and does not fabricate extracted values or answers.

When an HTTP adapter is available, configure `VITE_API_BASE_URL` and provide:

* `POST /api/bills/extract` — multipart field `file`; response should contain a bill object in `bill` or `extraction.bill`.
* `POST /api/bills/answer` — multipart fields `file` (optional), `question`, and `bill`; response should contain an `answer` string.

The browser never receives or stores a Gemini API key. With no base URL, uploads still preview locally and the UI clearly reports that live extraction/Q&A is unavailable.
