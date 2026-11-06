import { useMemo, useState } from 'react'

function historyGroup(timestamp) {
  const date = new Date(timestamp)
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  if (date >= today) return 'Today'
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  if (date >= yesterday) return 'Yesterday'
  const lastWeek = new Date(today)
  lastWeek.setDate(lastWeek.getDate() - 7)
  return date >= lastWeek ? 'Previous 7 days' : 'Earlier'
}

export default function Sidebar({ session, sessions, onSelect, onNew, busy, historyLoading, historyError, onRetryHistory }) {
  const [search, setSearch] = useState('')
  const groups = useMemo(() => {
    const filtered = sessions.filter(item => item.doc_name.toLowerCase().includes(search.trim().toLowerCase()))
    return ['Today', 'Yesterday', 'Previous 7 days', 'Earlier'].map(label => ({
      label, items: filtered.filter(item => historyGroup(item.updated_at) === label),
    })).filter(group => group.items.length)
  }, [sessions, search])

  return (
    <aside className="sidebar">
      <div className="sidebar-brand"><span className="brand-mark" aria-hidden="true">D</span>DocChat<span className="brand-ai">AI</span></div>
      <button className="new-chat-button" onClick={onNew} disabled={busy}><span aria-hidden="true">＋</span> New chat</button>
      <div className="history-heading"><h2>Your conversations</h2><span>{sessions.length}</span></div>
      <label className="history-search">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4 4" /></svg>
        <input type="search" aria-label="Search conversations" placeholder="Search chats…" value={search} onChange={e => setSearch(e.target.value)} />
      </label>
      <nav className="session-history" aria-label="Previous conversations">
        {groups.map(group => <section className="history-group" key={group.label}>
          <h3>{group.label}</h3>
          {group.items.map(item => <button key={item.session_id}
            className={`session-item${session?.session_id === item.session_id ? ' active' : ''}`}
            aria-current={session?.session_id === item.session_id ? 'page' : undefined}
            title={item.doc_name} disabled={busy}
            onClick={() => onSelect(item)}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M14 3H6v18h12V7zM14 3v5h4M9 12h6M9 16h4" /></svg>
            <span className="session-details"><span className="session-name">{item.doc_name}</span>
              <span className="session-date">{new Date(item.updated_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span>
            </span>
          </button>)}
        </section>)}
        {historyLoading && <p className="history-empty" role="status">Loading history…</p>}
        {!historyLoading && !groups.length && !historyError && <p className="history-empty">{search ? 'No conversations match your search.' : 'Your conversations will appear here after you upload a document.'}</p>}
      </nav>
      {historyError && <div className="history-error" role="alert"><p>{historyError}</p><button className="quiet-button" onClick={onRetryHistory} disabled={busy}>Retry history</button></div>}
      <div className="sidebar-footer"><span className="privacy-dot" /> Your chats, just for you.</div>
    </aside>
  )
}
