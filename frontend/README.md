# MONSOON / TWIN frontend

The dashboard sends forecasts and one-time perturbations to the backend's
verified primary GRaST bundle in `../artifacts/primary`. The backend health
response must report a model name beginning with `grast-primary-`; the UI
rejects persistence or other fallback engines instead of showing their output.

## Run locally

From the repository root, start the API:

```bash
python serve.py --port 8000
```

Then start Vite:

```bash
cd frontend
npm run dev
```

Vite proxies `/health` and `/simulate` to `http://localhost:8000`. Set
`VITE_BACKEND_URL` only when the API is hosted at a different origin.

The frontend does not load model files itself. The API loads and verifies the
primary bundle's `weights.pt`, manifest, and static features; if that load fails
and the API reports a fallback engine, the dashboard blocks simulation and
displays the error.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `tsrs` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
