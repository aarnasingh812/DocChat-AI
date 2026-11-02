import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, thinking, session, onSend }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  if (messages.length === 0 && !thinking) {
    return (
      <div className="chat-window document-welcome">
        <section className="document-ready" aria-labelledby="document-ready-title">
          <span className="ready-label"><span className="privacy-dot" /> Document ready</span>
          <h2 id="document-ready-title">Let’s explore your document</h2>
          <p className="ready-filename">{session.doc_name}</p>
          <p>Ask your first question below, or try one of these to get started.</p>
          <div className="suggestion-list">
            {['Summarize this document.', 'What are the key takeaways?', 'What are the most important details?'].map(question => (
              <button key={question} className="suggestion-button" onClick={() => onSend(question)}>{question}<span aria-hidden="true">↗</span></button>
            ))}
          </div>
        </section>
      </div>
    )
  }

  return (
    <div className="chat-window">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} msg={msg} />
      ))}

      {/* Typing indicator */}
      {thinking && (
        <div className="msg-row assistant" style={{ animation: 'fadeUp .3s ease both' }}>
          <div className="msg-avatar">🤖</div>
          <div className="msg-content">
            <div className="typing-indicator">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
