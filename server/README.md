# CardioFusion server

Create `.env` from `.env.example`, set the PostgreSQL `DATABASE_URL`, Clerk
secret, Cloudinary credentials, Gemini key, and `ML_SERVICE_URL`.

```powershell
npm install
npm run prisma:generate
npx prisma migrate dev --name init
npm run dev
```

For an already-created PostgreSQL database, apply the checked-in migration
non-interactively with `npm run prisma:deploy`.

The API runs at `http://localhost:4000`. It exposes `/health`, authenticated
`/api/profile`, and authenticated `/api/reports` endpoints. Reports upload to
Cloudinary, call the FastAPI service, then receive conservative Gemini
recommendations. Gemini never changes the ML probabilities.