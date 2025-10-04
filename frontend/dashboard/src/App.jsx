import { useState } from 'react'
import { searchContent, summarizeQuery, fetchAllPages } from './services/api'
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

  const handleSearch = async () => {
    setLoading(true)
    setError(null)
    setSearchResults([])
    setSummary('')
    
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
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search your browsing history... (e.g., 'state space search')"
                  className="search-input"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleSearch()
                    }
                  }}
                />
              </div>
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`filter-btn ${showFilters ? 'active' : ''}`}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"></polygon>
                </svg>
                Filters
              </button>
              <button
                onClick={handleSearch}
                disabled={loading}
                className="search-btn"
              >
                {loading ? (
                  <>
                    <svg className="spinner" width="16" height="16" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="31.416" strokeDashoffset="31.416">
                        <animate attributeName="stroke-dasharray" dur="2s" values="0 31.416;15.708 15.708;0 31.416" repeatCount="indefinite"/>
                        <animate attributeName="stroke-dashoffset" dur="2s" values="0;-15.708;-31.416" repeatCount="indefinite"/>
                      </circle>
                    </svg>
                    Searching...
                  </>
                ) : (
                  'Search'
                )}
              </button>
            </div>

            {/* Advanced Filters */}
            {showFilters && (
              <div className="filters-panel">
                <div className="filter-group">
                  <label className="filter-label">
                    <span className="filter-label-text">From Date</span>
                    <input
                      type="date"
                      value={dateFrom}
                      onChange={(e) => setDateFrom(e.target.value)}
                      className="filter-input"
                    />
                  </label>
                  <label className="filter-label">
                    <span className="filter-label-text">To Date</span>
                    <input
                      type="date"
                      value={dateTo}
                      onChange={(e) => setDateTo(e.target.value)}
                      className="filter-input"
                    />
                  </label>
                  <button
                    onClick={() => {
                      setDateFrom('')
                      setDateTo('')
                    }}
                    className="clear-btn"
                  >
                    Clear Dates
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="15" y1="9" x2="9" y2="15"></line>
                <line x1="9" y1="9" x2="15" y2="15"></line>
              </svg>
              {error}
            </div>
          )}

          {/* Summary */}
          {summary && (
            <div className="summary-card">
              <div className="summary-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14,2 14,8 20,8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10,9 9,9 8,9"></polyline>
                </svg>
                <h3>AI Summary</h3>
              </div>
              <p className="summary-text">{summary}</p>
            </div>
          )}

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="results-section">
              <div className="results-header">
                <h3>Search Results ({searchResults.length})</h3>
                <div className="results-info">
                  Found {searchResults.length} relevant content pieces
                </div>
              </div>
              
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
                        >
                          {result.title}
        </a>
                        {result.url && (
                          <div className="result-url">
                            <a href={result.url} target="_blank" rel="noopener noreferrer">
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

          {/* No Results */}
          {searchResults.length === 0 && !loading && !error && query && (
            <div className="no-results">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                <circle cx="11" cy="11" r="8"></circle>
                <path d="m21 21-4.35-4.35"></path>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
              </svg>
              <h3>No results found</h3>
              <p>Try adjusting your search terms or date filters</p>
            </div>
          )}
        </div>
      </main>
      </div>
  )
}

export default App