/**
 * Single source of truth for the backend base URL.
 * Set VITE_API_URL in frontend/.env (or .env.production) for non-localhost deployments.
 * Example:  VITE_API_URL=https://api.yourdomain.com
 */
export const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function apiFetch(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    credentials: 'include',
    headers: { 'X-DocChat-Request': '1', ...options.headers },
  })
  if (response.status === 401 && !path.startsWith('/auth/')) {
    window.dispatchEvent(new Event('docchat:unauthorized'))
  }
  return response
}

export async function authRequest(path, body) {
  const response = await apiFetch(`/auth/${path}`, body ? {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  } : {})
  const data = await response.json()
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(item => item.msg).join(' ')
      : data.detail
    throw new Error(detail || 'Unable to complete request. Please try again.')
  }
  return data
}
