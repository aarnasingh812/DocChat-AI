import { useState } from 'react'
import { authRequest } from '../api'

export default function AuthScreen({ onAuthenticated }) {
  const [signup, setSignup] = useState(false)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault()
    setError('')
    if (signup && password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }
    setBusy(true)
    try {
      const user = await authRequest(signup ? 'signup' : 'login', {
        email: email.trim(), password, ...(signup ? { name: name.trim() } : {}),
      })
      onAuthenticated(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  function switchMode() {
    setSignup(value => !value)
    setPassword('')
    setConfirmPassword('')
    setError('')
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="sidebar-brand">📄 DocChat AI</div>
        <h1 id="auth-title">{signup ? 'Create your account' : 'Welcome back'}</h1>
        <p className="auth-description">
          {signup ? 'Your documents, your questions, your workspace.' : 'Sign in to start chatting with your documents.'}
        </p>
        <form onSubmit={submit} className="auth-form">
          {signup && <label htmlFor="auth-name">Name
            <input id="auth-name" autoComplete="name" value={name} onChange={e => setName(e.target.value)} required maxLength={80} disabled={busy} />
          </label>}
          <label htmlFor="auth-email">Email address
            <input id="auth-email" type="email" autoComplete="email" value={email} onChange={e => setEmail(e.target.value)} required maxLength={254} disabled={busy} />
          </label>
          <label htmlFor="auth-password">Password
            <input id="auth-password" type="password" autoComplete={signup ? 'new-password' : 'current-password'} value={password} onChange={e => setPassword(e.target.value)} required minLength={signup ? 12 : 1} maxLength={128} disabled={busy} aria-describedby={signup ? 'password-hint' : undefined} />
          </label>
          {signup && <>
            <p id="password-hint" className="auth-hint">Use 12–128 characters. A long passphrase works well.</p>
            <label htmlFor="auth-confirm">Confirm password
              <input id="auth-confirm" type="password" autoComplete="new-password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required minLength={12} maxLength={128} disabled={busy} />
            </label>
          </>}
          {error && <div className="auth-error" role="alert">{error}</div>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy ? 'Please wait…' : signup ? 'Create account' : 'Sign in'}
          </button>
        </form>
        <p className="auth-switch">
          {signup ? 'Already have an account?' : 'New to DocChat?'}{' '}
          <button type="button" onClick={switchMode} disabled={busy}>{signup ? 'Sign in' : 'Sign up'}</button>
        </p>
      </section>
    </main>
  )
}
