import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import ChatWindow from './ChatWindow'
import ChatInput from './ChatInput'

// The parent keys this entire panel by session ID. Messages, errors, composer,
// loading state and source accordions belong to exactly one conversation.
export default function ConversationPanel({ document, onActivity, onDelete, disabled }) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(true)
  const [thinking, setThinking] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [retry, setRetry] = useState(0)
  const mounted = useRef(false)
  const sending = useRef(false)
  const busy = disabled || loading || thinking || deleting || Boolean(loadError)

  useEffect(() => {
    let active = true
    mounted.current = true
    setLoading(true)
    setLoadError(null)
    async function load() {
      try {
        const response = await apiFetch(`/session/${document.session_id}/messages`)
        if (!response.ok) throw new Error('Unable to open this conversation. Please try again.')
        const data = await response.json()
        if (active) setMessages(data.messages)
      } catch (err) {
        if (active) setLoadError(err.message)
      } finally {
        if (active) setLoading(false)
      }
    }
    load()
    return () => { active = false; mounted.current = false }
  }, [document.session_id, retry])

  async function send(question) {
    if (busy || sending.current) return
    sending.current = true
    setThinking(true)
    setError(null)
    const id = crypto.randomUUID()
    setMessages(prev => [...prev, { id, role: 'user', content: question }])
    try {
      const response = await apiFetch('/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: document.session_id, question }),
      })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Unable to send your message. Please try again.')
      }
      const data = await response.json()
      if (mounted.current) setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: 'assistant', content: data.answer,
        sources: data.sources, elapsed: data.elapsed,
      }])
      onActivity()
    } catch (err) {
      if (mounted.current) {
        setError(err.message)
        setMessages(prev => prev.filter(message => message.id !== id))
      }
    } finally {
      sending.current = false
      if (mounted.current) setThinking(false)
    }
  }

  async function remove() {
    if (busy || !window.confirm(`Delete the conversation with "${document.doc_name}" and its document? This cannot be undone.`)) return
    setDeleting(true)
    try { await onDelete(document) }
    catch (err) { if (mounted.current) setError(err.message) }
    finally { if (mounted.current) setDeleting(false) }
  }

  return (
    <section className="conversation-panel" aria-label={`Conversation with ${document.doc_name}`} data-session-id={document.session_id}>
      <div className="document-toolbar">
        <span className="document-chip"><span className="privacy-dot" /> PDF · {document.pages} {document.pages === 1 ? 'page' : 'pages'}</span>
        <button className="delete-chat-button" onClick={remove} disabled={busy}>Delete chat</button>
      </div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      {loading ? <div className="history-loading" role="status">Loading conversation…</div>
        : loadError ? <div className="conversation-load-error" role="alert"><p>{loadError}</p><button className="quiet-button" onClick={() => setRetry(value => value + 1)}>Try again</button></div>
          : <ChatWindow messages={messages} thinking={thinking} session={document} onSend={send} />}
      <ChatInput onSend={send} disabled={busy} />
    </section>
  )
}
