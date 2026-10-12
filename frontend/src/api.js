const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

async function request(path, options) {
  if (!API_BASE_URL) throw new Error('Backend API is not configured. Set VITE_API_BASE_URL to enable live extraction and Q&A.');
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) throw new Error(`Backend request failed (${response.status}).`);
  return response.json();
}

export async function extractBill(file) {
  if (!API_BASE_URL) return { source: 'unavailable', bill: {}, message: 'Backend API is not configured. Add VITE_API_BASE_URL, then upload again.' };
  const body = new FormData();
  body.append('file', file);
  return { source: 'backend', ...(await request('/api/bills/extract', { method: 'POST', body })) };
}

export async function askQuestion({ file, question, bill }) {
  if (!API_BASE_URL) return { source: 'unavailable', answer: 'The Q&A backend is not connected. Configure VITE_API_BASE_URL to ask grounded questions.' };
  const body = new FormData();
  if (file) body.append('file', file);
  body.append('question', question);
  body.append('bill', JSON.stringify(bill));
  return { source: 'backend', ...(await request('/api/bills/answer', { method: 'POST', body })) };
}
