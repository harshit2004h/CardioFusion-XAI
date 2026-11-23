# CardioFusion-XAI

CardioFusion-XAI is a research application with three services:

- **Client:** Next.js workspace for sign-in, evidence upload, report history, and assessment review.
- **Server:** Express API for Clerk authentication, PostgreSQL reports, Cloudinary uploads, and ML-service requests.
- **ML service:** FastAPI for biomarker, ECG, and echo processing, registry-based staged evidence evaluation, and per-modality explanations.

The browser sends authenticated requests to Express. Express stores uploaded assets in Cloudinary and sends HTTPS file references to FastAPI. FastAPI returns model outputs and disease results; Express stores the report in PostgreSQL; the client reads that report from Express. The client does not call the ML service directly.

The current application supports biomarkers, ECG, and echo. Evidence is combined through the configured staged registry rules; this repository does not contain a trained paired-modality fusion model. Outputs and explanations are research support, not clinical diagnoses.

## Run the application locally

Start PostgreSQL and make sure the configured Cloudinary account and Clerk application are available. Copy each example environment file and replace credentials with your own values. Keep secrets in service environment files; never put secret keys in `NEXT_PUBLIC_*` variables.

### 1. Start the ML service

```powershell
cd ml-service
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv-cardio\Scripts\Activate.ps1
python -m pip install -e .
python -m api.main
```

The ML service listens on `http://localhost:3002`. It needs the model checkpoints under `ml-service/checkpoints`; `/v1/health` reports which checkpoints actually loaded.

### 2. Start the Express server

In a second terminal:

```powershell
cd server
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run prisma:generate
npm run prisma:deploy
npm run dev
```

Set `DATABASE_URL`, `CLERK_SECRET_KEY`, and Cloudinary credentials in `server/.env`. `ML_SERVICE_URL` should point to `http://localhost:3002`. The default server port is `4000`.

### 3. Start the Next.js client

In a third terminal:

```powershell
cd client
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Set `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `NEXT_PUBLIC_SERVER_URL=http://localhost:4000` in `client/.env`. Open `http://localhost:3000` and sign in through Clerk.

## Integration checks

- Client: `cd client; npm run lint; npm run build`
- Server: `cd server; npm run build; npm test`
- ML: `cd ml-service; python -m pytest -q`
- Server liveness: `GET http://localhost:4000/health`
- ML liveness/readiness details: `GET http://localhost:3002/health` and `GET http://localhost:3002/v1/health`
- ML inference contract: `POST http://localhost:3002/v1/inference` with at least one modality containing an HTTPS `url`; WFDB ECG uses both `url` and `header_url`.

A successful build does not prove that a checkpoint, database, or external credential is available. Check `/v1/health` and the server logs before testing an authenticated report upload.

## Research and safety limits

The ML service preserves missing or unsupported modalities as such, and only validated evidence enters the disease registry. Gemini can suggest general next steps but cannot change model probabilities. SHAP and Integrated Gradients explain individual model behavior relative to their chosen baselines; they do not establish causation or clinical validity.
