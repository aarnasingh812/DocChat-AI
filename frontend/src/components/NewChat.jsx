import { useRef, useState } from 'react'
import { apiFetch } from '../api'

export default function NewChat({ onUpload, busy, onUploadingChange }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)
  const uploadLock = useRef(false)

  async function upload(file) {
    if (!file || busy || uploadLock.current) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please choose a PDF document.')
      if (inputRef.current) inputRef.current.value = ''
      return
    }
    uploadLock.current = true
    setUploading(true)
    onUploadingChange(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await apiFetch('/upload', { method: 'POST', body: form })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Unable to upload this document. Please try again.')
      }
      onUpload(await response.json())
    } catch (err) {
      setError(err.message)
    } finally {
      uploadLock.current = false
      setUploading(false)
      onUploadingChange(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div className="new-chat-view">
      <section className="new-chat-content" aria-labelledby="new-chat-title">
        <div className="welcome-symbol" aria-hidden="true">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8M8 16h5" /></svg>
        </div>
        <span className="welcome-eyebrow">A little less reading. A lot more clarity.</span>
        <h1 id="new-chat-title">What would you like to explore?</h1>
        <p className="welcome-description">Upload a document to start a new conversation.</p>

        <div className={`upload-card${dragging ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true) }}
          onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false) }}
          onDrop={e => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files[0]) }}
          aria-busy={uploading}>
          <input ref={inputRef} type="file" accept=".pdf,application/pdf" aria-label="Choose a PDF document" hidden disabled={busy} onChange={e => upload(e.target.files[0])} />
          <div className="upload-card-icon" aria-hidden="true">
            {uploading ? <span className="upload-spinner" /> : <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5" /></svg>}
          </div>
          <h2>{uploading ? 'Getting your document ready…' : 'Add your document'}</h2>
          <p>{uploading ? 'Reading and indexing your PDF. This may take a moment.' : 'Drop a PDF here or upload one to get started.'}</p>
          <button className="upload-button" onClick={() => inputRef.current?.click()} disabled={busy}>
            {uploading ? 'Processing PDF…' : 'Upload PDF'}<span aria-hidden="true">↗</span>
          </button>
          <span className="upload-note" role="status">{uploading ? 'Your new conversation will open automatically.' : 'PDF documents · Saved to your private chat history'}</span>
          {error && <div className="upload-error" role="alert">{error}</div>}
        </div>

        <ol className="welcome-steps" aria-label="How it works">
          <li><span className="step-num">1</span><div><strong>Upload a PDF</strong><p>Pick the document you want to explore.</p></div></li>
          <li><span className="step-num">2</span><div><strong>Ask anything</strong><p>Get clear answers with page references.</p></div></li>
          <li><span className="step-num">3</span><div><strong>Pick up later</strong><p>Reopen any chat from your history.</p></div></li>
        </ol>
      </section>
    </div>
  )
}
