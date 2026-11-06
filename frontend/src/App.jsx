import { useState, useCallback, useEffect, useRef } from 'react'
import Sidebar from './components/Sidebar'
import ConversationPanel from './components/ConversationPanel'
import NewChat from './components/NewChat'
import { apiFetch } from './api'
import AuthScreen from './components/AuthScreen'

export default function App() {
  const [user, setUser] = useState(null)
  const [checking, setChecking] = useState(true)
  const [authError, setAuthError] = useState('')
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    let active = true
    const expire = () => setUser(null)
    window.addEventListener('docchat:unauthorized', expire)
    async function restore() {
      setChecking(true)
      setAuthError('')
      try {
        const response = await apiFetch('/auth/me')
        if (response.status === 401) return
        if (!response.ok) throw new Error('Unable to connect to your account. Please try again.')
        const data = await response.json()
        if (active) setUser(data)
      } catch (err) {
        if (active) setAuthError(err.message)
      } finally {
        if (active) setChecking(false)
      }
    }
    restore()
    return () => {
      active = false
      window.removeEventListener('docchat:unauthorized', expire)
    }
  }, [retry])

  if (checking) return <main className="auth-page" role="status">Loading your workspace…</main>
  if (authError) return <main className="auth-page"><div className="auth-card" role="alert">
    <p>{authError}</p><button className="auth-submit" onClick={() => setRetry(value => value + 1)}>Try again</button>
  </div></main>
  if (!user) return <AuthScreen onAuthenticated={setUser} />
  return <Workspace key={user.id} user={user} onLogout={() => setUser(null)} />
}

export function Workspace({ user, onLogout }) {
  const [session, setSession] = useState(null)
  const [sessions, setSessions] = useState([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const [error, setError] = useState(null)
  const mounted = useRef(false)
  const historyRequest = useRef(0)
  const busy = uploading || signingOut

  const refreshHistory = useCallback(async () => {
    const version = ++historyRequest.current
    try {
      const response = await apiFetch('/sessions')
      if (!response.ok) throw new Error('Unable to load your chat history.')
      const data = await response.json()
      if (mounted.current && version === historyRequest.current) {
        setSessions(data)
        setHistoryError(null)
      }
    } catch (err) {
      if (mounted.current && version === historyRequest.current) setHistoryError(err.message)
    } finally {
      if (mounted.current && version === historyRequest.current) setHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    // Load sidebar history only. Every sign-in starts on the new-chat screen.
    refreshHistory()
    return () => { mounted.current = false; historyRequest.current += 1 }
  }, [refreshHistory])

  const handleNew = useCallback(() => {
    setSession(null)
    setError(null)
  }, [])

  const handleUpload = useCallback((document) => {
    setSession(document)
    setError(null)
    refreshHistory()
  }, [refreshHistory])

  const handleDelete = useCallback(async (document) => {
    const response = await apiFetch(`/session/${document.session_id}`, { method: 'DELETE' })
    if (!response.ok) throw new Error('Unable to delete this conversation. Please try again.')
    if (!mounted.current) return
    setSessions(prev => prev.filter(item => item.session_id !== document.session_id))
    setSession(current => current?.session_id === document.session_id ? null : current)
    refreshHistory()
  }, [refreshHistory])

  async function signOut() {
    setSigningOut(true)
    setError(null)
    try {
      const response = await apiFetch('/auth/logout', { method: 'POST' })
      if (!response.ok) throw new Error('Unable to sign out. Please try again.')
      onLogout()
    } catch (err) {
      if (mounted.current) setError(err.message)
    } finally {
      if (mounted.current) setSigningOut(false)
    }
  }

  return (
    <>
      <Sidebar session={session} sessions={sessions}
        onSelect={document => { setSession(document); setError(null) }} onNew={handleNew}
        busy={busy} historyLoading={historyLoading} historyError={historyError} onRetryHistory={refreshHistory} />
      <main className="main">
        <header className="topbar">
          <div className="conversation-title">
            <span className="topbar-title">{session ? session.doc_name : 'New chat'}</span>
            <span className="topbar-sub">DocChat AI</span>
          </div>
          <div className="account-controls">
            <span className="account-name" title={user.email}>{user.name}</span>
            <button className="quiet-button" onClick={signOut} disabled={busy}>{signingOut ? 'Signing out…' : 'Sign out'}</button>
          </div>
        </header>
        {error && <div className="error-banner" role="alert">{error}</div>}
        {session
          ? <ConversationPanel key={session.session_id} document={session} onActivity={refreshHistory} onDelete={handleDelete} disabled={busy} />
          : <NewChat onUpload={handleUpload} busy={busy} onUploadingChange={setUploading} />}
      </main>
    </>
  )
}
