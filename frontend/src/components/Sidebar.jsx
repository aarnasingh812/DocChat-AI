import { useState, useRef, useCallback } from 'react'
import { API } from '../api'

export default function Sidebar({ session, onUpload, onClear }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported.')
      return
    }
    setError(null)
    setUploading(true)

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed.')
      }
      const data = await res.json()
      onUpload(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }, [onUpload])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <span className="sidebar-brand-icon">📄</span>
        DocChat AI
      </div>
      

      <div className="sidebar-divider" />

      {/* Upload */}
      <div className="sidebar-section-title">Upload Document</div>
      <div
        className={`drop-zone${dragging ? ' drag-over' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !uploading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          onChange={(e) => { setError(null); handleFile(e.target.files[0]) }}
          disabled={uploading}
          style={{ display: 'none' }}
        />
        <span className="drop-zone-icon">{uploading ? '⏳' : '📂'}</span>
        <div className="drop-zone-text">
          {uploading
            ? 'Processing document…'
            : (<><strong>Click to upload</strong> or drag & drop<br />PDF files only</>)
          }
        </div>
      </div>

      {/* Upload progress */}
      {uploading && (
        <div className="upload-progress">
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: '100%' }} />
          </div>
          <div className="progress-label">
            <div className="spinner-xs" />
            Embedding document chunks…
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ marginTop: 10, fontSize: '.75rem', color: 'var(--clr-error)', display: 'flex', gap: 6, alignItems: 'flex-start' }}>
          <span>⚠️</span> {error}
        </div>
      )}

      {/* Doc info */}
      {session && (
        <>
          <div className="sidebar-divider" />
          <div className="sidebar-section-title">Document Info</div>
          <div className="doc-card">
            <div className="doc-card-name">
              <span>📄</span>
              {session.doc_name.length > 28
                ? session.doc_name.slice(0, 28) + '…'
                : session.doc_name}
            </div>
            <div className="doc-stats">
              <div className="stat-box">
                <div className="stat-box-val">{session.pages}</div>
                <div className="stat-box-label">Pages</div>
              </div>
              <div className="stat-box">
                <div className="stat-box-val">{session.chunks}</div>
                <div className="stat-box-label">Chunks</div>
              </div>
            </div>
          </div>

          <div className="status-ready" style={{ marginTop: 12 }}>
            <span className="status-dot" />
            Ready to chat
          </div>
        </>
      )}

      {/* Clear */}
      {session && (
        <>
          <div className="sidebar-divider" />
          <button className="btn-clear" onClick={onClear}>
            🗑️ Clear Chat &amp; Reset
          </button>
        </>
      )}

    </aside>
  )
}
