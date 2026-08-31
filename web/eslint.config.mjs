import { defineConfig, globalIgnores } from "eslint/config";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores([
    "dist/**",
    ".vinext/**",
    ".next/**",
    "out/**",
    "build/**",
    "node_modules/**",
  ]),
  ...tseslint.configs.recommended,
]);
