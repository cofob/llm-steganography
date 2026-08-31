# LLM Steganography Web

This is the Vinext and Vite client for the project Litestar API. The interface uses
vendored `@cofob/design-system-css` and `@cofob/design-system-react` version 0.5.0.

## Run

Start the API from the repository root:

```bash
uv run llm-steg-api
```

Start the web application:

```bash
cd web
npm install
npm run dev
```

`NEXT_PUBLIC_STEG_API_URL` sets the API address. The default value is
`http://127.0.0.1:8000`. See `.env.example`.

## Check

```bash
npm run lint
npm run typecheck
npm run build
```

The web runtime is `vinext@1.0.0-beta.8`. The standard `dev`, `build`, and `start`
commands use vinext instead of the Next.js CLI.
