import { useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import { API } from './api'

export default function App() {
  const [session, setSession]   = useState(null)   // { session_id, doc_name, pages, chunks }
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false)
  const [error, setError]       = useState(null)

  // Called by Sidebar once upload succeeds
  const handleUpload = useCallback((data) => {
    setSession(data)
    setMessages([])
    setError(null)
  }, [])

  // Clear chat + session
  const handleClear = useCallback(async () => {
    if (session?.session_id) {
      try {
        await fetch(`${API}/session/${session.session_id}`, { method: 'DELETE' })
      } catch (_) { /* best-effort */ }
    }
    setSession(null)
    setMessages([])
    setError(null)
  }, [session])

  // Send a message through the RAG pipeline
  const handleSend = useCallback(async (question) => {
    if (!session?.session_id) return
    setError(null)

    // Optimistic user message
    const msgId = crypto.randomUUID()
    setMessages(prev => [...prev, { id: msgId, role: 'user', content: question }])
    setThinking(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: session.session_id, question }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Chat request failed.')
      }

      const data = await res.json()
      setMessages(prev => [
        ...prev,
        {
          id:      crypto.randomUUID(),
          role:    'assistant',
          content: data.answer,
          sources: data.sources,
          elapsed: data.elapsed,
        },
      ])
    } catch (e) {
      setError(e.message)
      // Remove the optimistic user bubble if the request failed
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setThinking(false)
    }
  }, [session])

  return (
    <>
      <Sidebar
        session={session}
        onUpload={handleUpload}
        onClear={handleClear}
      />

      <main className="main">
        {/* Top bar */}
        <header className="topbar">
          <div>
            <div className="topbar-title">Chat with Your Doc</div>
            <div className="topbar-sub">
              {session
                ? `Chatting with ${session.doc_name}`
                : 'Upload a PDF in the sidebar to get started'}
            </div>
          </div>
          <div className="topbar-badge">DocChat AI</div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="error-banner">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Chat area */}
        <ChatWindow messages={messages} thinking={thinking} />

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          disabled={!session || thinking}
        />
      </main>
    </>
  )
}
