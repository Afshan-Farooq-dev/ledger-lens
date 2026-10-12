import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowUpRight, Check, ChevronDown, Copy, FileJson, FileText, Info,
  LoaderCircle, MessageCircle, MoreHorizontal, Paperclip, Play, Plus,
  RefreshCw, Send, Sparkles, UploadCloud, X, ZoomIn
} from 'lucide-react';
import './styles.css';
import { askQuestion, extractBill } from './api.js';

const initialWorkflow = [
  { id: 'upload', title: 'Bill upload', detail: 'Waiting for a document', state: 'active', icon: UploadCloud },
  { id: 'prep', title: 'Preprocessing', detail: 'OpenCV enhancement', state: 'waiting', icon: Sparkles },
  { id: 'ocr', title: 'OCR pass', detail: 'Tesseract text layer', state: 'waiting', icon: FileText },
  { id: 'vision', title: 'Vision model', detail: 'Gemini 3.1 Flash-Lite', state: 'waiting', icon: Sparkles },
  { id: 'validate', title: 'Validation', detail: 'Python evidence checks', state: 'waiting', icon: Check },
  { id: 'json', title: 'Level 1 JSON', detail: 'Structured bill record', state: 'waiting', icon: FileJson },
  { id: 'qa', title: 'Ask your bill', detail: 'Grounded Q&A ready', state: 'waiting', icon: MessageCircle }
];

const demoExtraction = {
  provider: 'Awaiting backend',
  tariff: '—',
  billing_period: '—',
  units_consumed: null,
  current_bill: null,
  total_payable: null,
  due_date: '—',
  status: 'No extraction yet'
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [workflow, setWorkflow] = useState(initialWorkflow);
  const [extraction, setExtraction] = useState(demoExtraction);
  const [jsonOpen, setJsonOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [zoom, setZoom] = useState(false);

  const setNode = (id, state) => setWorkflow((items) => items.map((item) => item.id === id ? { ...item, state } : item));

  const handleFile = async (nextFile) => {
    if (!nextFile) return;
    if (!['image/jpeg', 'image/png'].includes(nextFile.type)) {
      setNotice('Please choose a JPG, JPEG, or PNG bill image.');
      return;
    }
    setNotice('');
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setWorkflow(initialWorkflow.map((node) => node.id === 'upload' ? { ...node, state: 'complete', detail: 'Document received' } : node));
    setBusy(true);
    try {
      for (const id of ['prep', 'ocr', 'vision', 'validate', 'json', 'qa']) setNode(id, 'active');
      const result = await extractBill(nextFile);
      const nextBill = result.bill || result.extraction?.bill;
      if (nextBill && Object.keys(nextBill).length > 0) {
        setExtraction(nextBill);
        setWorkflow(initialWorkflow.map((node) => ({ ...node, state: 'complete' })));
      } else {
        setWorkflow(initialWorkflow.map((node) => node.id === 'upload' ? { ...node, state: 'complete', detail: 'Document received' } : { ...node, state: 'waiting' }));
      }
      setNotice(result.source === 'backend' ? 'Extraction complete.' : result.message || 'Backend is not connected yet.');
    } catch (error) {
      setWorkflow((items) => items.map((node) => node.id === 'upload' ? node : { ...node, state: 'failed' }));
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  };

  const submitQuestion = async (event) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    setQuestion('');
    setMessages((items) => [...items, { role: 'user', text }]);
    setBusy(true);
    try {
      const result = await askQuestion({ file, question: text, bill: extraction });
      setMessages((items) => [...items, { role: 'assistant', text: result.answer, grounded: result.source === 'backend' }]);
    } catch (error) {
      setMessages((items) => [...items, { role: 'assistant', text: error.message, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const json = JSON.stringify(extraction, null, 2);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top"><span className="brand-mark">L</span><span>ledger<span className="muted">/</span>decode</span></a>
        <div className="topbar-meta"><span className="status-dot" /> Local workspace <span className="divider" /> <span className="mono">v0.1</span><button className="icon-button"><MoreHorizontal size={18} /></button></div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">UTILITY INTELLIGENCE / 01</p>
          <h1>Make every line<br /><em>accountable.</em></h1>
          <p className="hero-copy">Upload a utility bill, inspect the evidence, and ask questions in plain language. Every answer is grounded in the bill record.</p>
        </div>
        <div className="hero-aside"><span className="mono">SESSION</span><strong>{file ? 'DOCUMENT LOADED' : 'READY TO BEGIN'}</strong><span>Level 1 extraction + Level 2 Q&A</span></div>
      </section>

      <section className="workspace-grid">
        <div className="primary-column">
          <div className="section-heading"><div><span className="section-index">01</span><h2>Document intake</h2></div><span className="caption">JPG / JPEG / PNG · max 10 MB</span></div>
          <label className={`dropzone ${file ? 'has-file' : ''}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files[0]); }}>
            <input type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(event) => handleFile(event.target.files[0])} />
            {preview ? <img src={preview} alt="Uploaded utility bill preview" /> : <><span className="upload-icon"><UploadCloud size={25} /></span><strong>Drop your bill here</strong><span>or <u>browse files</u> from your computer</span></>}
            {preview && <span className="file-overlay"><Check size={15} /> {file.name}</span>}
          </label>
          {notice && <div className="notice"><Info size={16} /> {notice}</div>}

          <div className="section-heading workflow-heading"><div><span className="section-index">02</span><h2>Processing workflow</h2></div><button className="text-button"><RefreshCw size={14} /> Replay</button></div>
          <div className="workflow-card">
            {workflow.map((node, index) => <React.Fragment key={node.id}>
              <div className={`workflow-node ${node.state}`}><div className="node-icon">{node.state === 'active' ? <LoaderCircle className="spin" size={17} /> : <node.icon size={17} />}</div><div><strong>{node.title}</strong><span>{node.detail}</span></div><span className="node-state">{node.state}</span></div>
              {index < workflow.length - 1 && <div className={`connector ${workflow[index + 1].state === 'active' || workflow[index + 1].state === 'complete' ? 'flowing' : ''}`}><span /></div>}
            </React.Fragment>)}
          </div>

          <div className="section-heading"><div><span className="section-index">03</span><h2>Ask your bill</h2></div><span className="caption">Grounded responses only</span></div>
          <div className="chat-card">
            <div className="chat-header"><div><strong>Bill analyst</strong><span>Answers are generated from extracted evidence</span></div><span className="live-pill"><span /> READY</span></div>
            <div className="messages">
              {!messages.length && <div className="empty-chat"><MessageCircle size={22} /><p>Ask anything about this bill</p><span>Try “How much tax was charged?” or “What is my due date?”</span></div>}
              {messages.map((message, index) => <div className={`message ${message.role} ${message.error ? 'error' : ''}`} key={index}><span className="message-label">{message.role === 'user' ? 'YOU' : 'ANALYST'}</span><p>{message.text}</p>{message.grounded && <small><Check size={12} /> Grounded in bill evidence</small>}</div>)}
              {busy && <div className="typing"><span /><span /><span /></div>}
            </div>
            <form className="chat-input" onSubmit={submitQuestion}><Paperclip size={17} /><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your bill..." /><button type="submit" aria-label="Send question"><Send size={17} /></button></form>
          </div>
        </div>

        <aside className="sidebar">
          <div className="section-heading"><div><span className="section-index">A</span><h2>Bill preview</h2></div><button className="icon-button" onClick={() => setZoom(!zoom)}><ZoomIn size={16} /></button></div>
          <div className={`preview-card ${zoom ? 'zoomed' : ''}`}>{preview ? <img src={preview} alt="Bill preview" /> : <div className="preview-placeholder"><FileText size={30} /><span>Preview appears here</span><small>Upload a bill to inspect it</small></div>}{file && <div className="preview-footer"><span className="truncate">{file.name}</span><span>{(file.size / 1024).toFixed(0)} KB</span></div>}</div>

          <div className="section-heading extraction-heading"><div><span className="section-index">B</span><h2>Extracted record</h2></div><button className="icon-button" onClick={() => navigator.clipboard?.writeText(json)} title="Copy JSON"><Copy size={15} /></button></div>
          <div className="record-card">
            {[['Provider', extraction.provider], ['Tariff', extraction.tariff], ['Billing period', extraction.billing_period], ['Units consumed', extraction.units_consumed ?? '—'], ['Current bill', extraction.current_bill ?? '—'], ['Due date', extraction.due_date]].map(([label, value]) => <div className="record-row" key={label}><span>{label}</span><strong>{String(value)}</strong></div>)}
            <button className="json-toggle" onClick={() => setJsonOpen(!jsonOpen)}><FileJson size={15} /> {jsonOpen ? 'Hide JSON' : 'View full JSON'} <ChevronDown size={14} className={jsonOpen ? 'rotate' : ''} /></button>
            {jsonOpen && <pre className="json-viewer">{json}</pre>}
          </div>
          <div className="privacy-note"><Info size={14} /><span>Bill images stay in your local session. API keys never enter the browser.</span></div>
        </aside>
      </section>
      <footer><span>LEDGER / DECODE</span><span>Built for transparent utility intelligence <ArrowUpRight size={13} /></span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
