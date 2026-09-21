import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, thinking }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  if (messages.length === 0 && !thinking) {
    return (
      <div className="chat-window welcome">
        <div className="welcome-card">
          <span className="welcome-icon">📄</span>
          <h2>No document loaded yet</h2>
          <p>
            Upload a PDF in the sidebar to start an intelligent conversation
            with your document.
          </p>
          <div className="welcome-steps">
            <div className="welcome-step">
              <div className="step-num">1</div>
              Click <strong>&nbsp;Upload Document&nbsp;</strong> in the sidebar to choose a PDF.
            </div>
            <div className="welcome-step">
              <div className="step-num">2</div>
              Wait a moment while the document is embedded and indexed.
            </div>
            <div className="welcome-step">
              <div className="step-num">3</div>
              Type your question below and get instant, cited answers.
            </div>
          </div>
        </div>
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
