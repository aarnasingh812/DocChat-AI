import { useState, useRef, useCallback } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const autoResize = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  const handleChange = (e) => {
    setValue(e.target.value)
    autoResize()
  }

  const submit = useCallback(() => {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [value, disabled, onSend])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="chat-input-area">
      <div className="chat-input-wrap">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={1}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          aria-label="Message this document"
          placeholder="Ask anything about your document"
          id="chat-input"
        />
        <button
          className="btn-send"
          onClick={submit}
          disabled={disabled || !value.trim()}
          title="Send message"
          id="send-btn"
          aria-label="Send message"
        >
          ➤
        </button>
      </div>
      <div className="chat-hint">
        Press <kbd style={{ fontFamily: 'var(--font-mono)', fontSize: '.65rem', padding: '1px 4px', background: 'var(--clr-bg-hover)', borderRadius: 3, border: '1px solid var(--clr-border)' }}>Enter</kbd> to send &nbsp;·&nbsp;
        <kbd style={{ fontFamily: 'var(--font-mono)', fontSize: '.65rem', padding: '1px 4px', background: 'var(--clr-bg-hover)', borderRadius: 3, border: '1px solid var(--clr-border)' }}>Shift+Enter</kbd> for newline
      </div>
    </div>
  )
}
