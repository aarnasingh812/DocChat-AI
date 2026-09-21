/**
 * Single source of truth for the backend base URL.
 * Set VITE_API_URL in frontend/.env (or .env.production) for non-localhost deployments.
 * Example:  VITE_API_URL=https://api.yourdomain.com
 */
export const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
