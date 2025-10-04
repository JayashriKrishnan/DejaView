import { useState } from 'react'
import { searchContent, summarizeQuery, fetchAllPages, getRagAnswer } from './services/api'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [summary, setSummary] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [ragAnswer, setRagAnswer] = useState(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [showRagOnly, setShowRagOnly] = useState(false)
  const [showDateFilter, setShowDateFilter] = useState(false)

  const handleSearch = async () => {
    setLoading(true)
    setError(null)
    setSearchResults([])
    setSummary('')
    setRagAnswer(null)
    setShowRagOnly(false)
    try {
      const results = await searchContent(
        query,
        50,         // top_k
        0.0,        // min_similarity
        dateFrom || undefined,
        dateTo || undefined
      )
      setSearchResults(Array.isArray(results.results) ? results.results : results)
      const nonEmpty = Array.isArray(results.results) ? results.results : results
      if (nonEmpty.length > 0) {
        const summaryResult = await summarizeQuery(query)
        setSummary(summaryResult.summary)
      }
    } catch (err) {
      setError('Failed to perform search. Please make sure your backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleRagAnswer = async () => {
    setRagLoading(true)
    setRagAnswer(null)
    setShowRagOnly(true)
    try {
      const result = await getRagAnswer(query, 5)
      setRagAnswer(result)
    } catch (err) {
      setRagAnswer({ ai_answer: 'Error generating answer.' })
    } finally {
      setRagLoading(false)
    }
  }

  const formatDate = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getSimilarityColor = (score) => {
    if (score >= 0.8) return '#10b981' // green
    if (score >= 0.6) return '#f59e0b' // yellow
    return '#ef4444' // red
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🔍</span>
            <h1 className="logo-text">DejaView</h1>
          </div>
          <p className="header-subtitle">Intelligent browsing history search & analysis</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="search-container">
          {/* Search Bar */}
          <div className="search-section">
            <div className="search-bar">
              <div className="search-input-container">
                <svg className="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"></circle>
                  <path d="m21 21-4.35-4.35"></path>
                </svg>
                <input
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Search your browsing history..."
                  className="search-input"
                />
                <button onClick={handleSearch} disabled={loading || !query} className="btn">{loading ? 'Searching...' : 'Search'}</button>
                <button onClick={handleRagAnswer} disabled={ragLoading || !query} className="btn" style={{marginLeft: '10px'}}>{ragLoading ? 'Generating AI Answer...' : 'Get AI Answer'}</button>
                <button className="filter-btn" onClick={() => setShowDateFilter(v => !v)}>
                  {showDateFilter ? 'Hide Filters' : 'Filter by Date'}
                </button>
              </div>
            </div>
            {showDateFilter && (
              <div className="date-filter-container">
                <label style={{marginRight: '10px'}}>
                  From:
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={e => setDateFrom(e.target.value)}
                    className="date-input"
                  />
                </label>
                <label>
                  To:
                  <input
                    type="date"
                    value={dateTo}
                    onChange={e => setDateTo(e.target.value)}
                    className="date-input"
                  />
                </label>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="results-section">
            {error && <div className="error-message">{error}</div>}
            {!showRagOnly && searchResults.length > 0 && (
              <div>
                <h2>Search Results</h2>
                <div className="results-grid">
                  {searchResults.map((result, index) => (
                    <div key={result.paragraph_id} className="result-card">
                      <div className="result-header">
                        <div className="result-title">
                          <a 
                            href={result.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="result-link"
                            style={{
                              maxWidth: '320px',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              display: 'inline-block',
                              verticalAlign: 'middle'
                            }}
                            title={result.title}
                          >
                            {result.title}
                          </a>
                          {result.url && (
                            <div className="result-url" style={{maxWidth: '320px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                              <a href={result.url} target="_blank" rel="noopener noreferrer" title={result.url} style={{color: '#2563eb', textDecoration: 'underline'}}>
                                {result.url}
                              </a>
                            </div>
                          )}
                        </div>
                        <div 
                          className="similarity-badge"
                          style={{ backgroundColor: getSimilarityColor(result.similarity_score) }}
                        >
                          {Math.round(result.similarity_score * 100)}% match
                        </div>
                      </div>
                      
                      <div className="result-content">
                        <p className="result-text">{result.text}</p>
                      </div>
                      
                      <div className="result-footer">
                        <div className="result-meta">
                          <span className="result-date">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                              <line x1="16" y1="2" x2="16" y2="6"></line>
                              <line x1="8" y1="2" x2="8" y2="6"></line>
                              <line x1="3" y1="10" x2="21" y2="10"></line>
                            </svg>
                            {formatDate(result.timestamp)}
                          </span>
                          <span className="result-paragraph">
                            Paragraph {result.paragraph_index + 1}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!showRagOnly && summary && (
              <div className="summary-section">
                <h3>Summary</h3>
                <p>{summary}</p>
              </div>
            )}
            {showRagOnly && ragAnswer && (
              <div className="rag-answer-section">
                <h3>🤖 AI Answer:</h3>
                <div style={{whiteSpace: 'pre-line', wordBreak: 'break-word', fontSize: '1.1rem', color: '#374151'}}>
                  {ragAnswer.ai_answer.replace(/\.\.\./g, '').replace(/\s+/g, ' ').trim()}
                </div>
                <h4>📚 Sources:</h4>
                <ul>
                  {ragAnswer.sources && ragAnswer.sources.map((src, idx) => (
                    <li key={idx} style={{maxWidth: '420px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                      <a href={src} target="_blank" rel="noopener noreferrer" title={src} style={{color: '#2563eb', textDecoration: 'underline'}}>
                        {src}
                      </a>
                    </li>
                  ))}
                </ul>
                <h4>💡 Key Takeaway:</h4>
                <div style={{whiteSpace: 'pre-line', wordBreak: 'break-word', fontSize: '1.05rem', color: '#059669'}}>
                  {ragAnswer.key_takeaway}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App