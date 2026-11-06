import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function MessageBubble({ msg }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = msg.role === 'user'

  return (
    <div className={`msg-row ${msg.role}`}>
      <div className="msg-avatar">
        {isUser ? 'You' : 'D'}
      </div>

      <div className="msg-content">
        <div className="msg-bubble">
          {isUser
            ? msg.content
            : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            )
          }
        </div>

        {/* Response meta */}
        {msg.elapsed != null && (
          <div className="msg-meta">
            <span>⚡</span>
            {msg.elapsed.toFixed(2)}s response time
          </div>
        )}

        {/* Sources accordion — assistant messages only */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <>
            <button
              className={`sources-toggle${sourcesOpen ? ' open' : ''}`}
              onClick={() => setSourcesOpen(v => !v)}
            >
              <span>📚 {msg.sources.length} source chunk{msg.sources.length !== 1 ? 's' : ''} used</span>
              <span className="sources-toggle-chevron">▼</span>
            </button>

            {sourcesOpen && (
              <div className="sources-list">
                {msg.sources.map((src, i) => (
                  <div className="source-chip" key={i}>
                    <div className="source-chip-header">
                      Chunk {i + 1}
                      {src.page != null ? ` · Page ${src.page}` : ''}
                    </div>
                    <div className="source-chip-text">{src.excerpt}</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
