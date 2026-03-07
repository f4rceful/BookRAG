import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search as SearchIcon, MessageSquare, BookOpen, MapPin, Sparkles, Filter, ArrowRight, ExternalLink, Info } from 'lucide-react';
import { Link } from 'react-router-dom';
import './Search.css';

interface SearchResult {
  content: string;
  source: string;
  location?: string;
  score: number;
  tome?: string;
  part?: string;
  chapter?: string;
  epilogue?: string;
}

interface AskResultData {
  answer: string;
  sources: SearchResult[];
}

const BOOK_QUERIES: Record<string, Array<{ text: string; type: string }>> = {
  'Булгаков — «Мастер и Маргарита».txt': [
    { text: 'Кто такой Воланд и зачем он приехал в Москву?', type: 'qa' },
    { text: 'Что произошло с Берлиозом у Патриарших прудов?', type: 'qa' },
    { text: 'Найди описание бала у Сатаны', type: 'search' },
  ],
  'Достоевский — «Преступление и наказание».txt': [
    { text: 'Почему Раскольников убил старуху-процентщицу?', type: 'qa' },
    { text: 'Как Соня Мармеладова повлияла на Раскольникова?', type: 'qa' },
    { text: 'Найди сцену признания Раскольникова Соне', type: 'search' },
  ],
  'Пушкин — «Евгений Онегин».txt': [
    { text: 'Чем закончилась история Онегина и Татьяны?', type: 'qa' },
    { text: 'Почему Онегин убил Ленского на дуэли?', type: 'qa' },
    { text: 'Найди письмо Татьяны к Онегину', type: 'search' },
  ],
  'Чехов — «Вишнёвый сад».txt': [
    { text: 'Кто купил вишнёвый сад и чем это закончилось?', type: 'qa' },
    { text: 'Почему Раневская не смогла спасти имение?', type: 'qa' },
    { text: 'Найди монолог Лопахина о покупке сада', type: 'search' },
  ],
};

const DEFAULT_QUERIES = [
  { text: 'Кто такой Воланд и зачем он приехал в Москву?', type: 'qa' },
  { text: 'Почему Раскольников убил старуху-процентщицу?', type: 'qa' },
  { text: 'Найди письмо Татьяны к Онегину', type: 'search' },
  { text: 'Кто купил вишнёвый сад и чем это закончилось?', type: 'qa' },
];

const Search = () => {
  const [activeTab, setActiveTab] = useState<'search' | 'qa'>('qa');

  // Состояние фильтра книг
  const [books, setBooks] = useState<string[]>([]);
  const [selectedBooks, setSelectedBooks] = useState<string[]>([]);

  // Состояние поиска
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchError, setSearchError] = useState('');

  // Состояние вопросов и ответов
  const [askQuery, setAskQuery] = useState('');
  const [askResult, setAskResult] = useState<AskResultData | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [askError, setAskError] = useState('');

  const exampleQueries =
    selectedBooks.length === 1
      ? (BOOK_QUERIES[selectedBooks[0]] ?? DEFAULT_QUERIES)
      : DEFAULT_QUERIES;

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/api/books`)
      .then((r) => r.json())
      .then((data) => setBooks(data.books || []))
      .catch(() => {});
  }, []);

  const toggleBook = (book: string) => {
    setSelectedBooks((prev) =>
      prev.includes(book) ? prev.filter((b) => b !== book) : [...prev, book]
    );
  };

  const activeSources = selectedBooks.length > 0 ? selectedBooks : undefined;

  const handleQuickQuery = (query: string, type: 'qa' | 'search') => {
    setActiveTab(type);
    if (type === 'qa') {
      setAskQuery(query);
      setTimeout(() => document.getElementById('ask-btn')?.click(), 100);
    } else {
      setSearchQuery(query);
      setTimeout(() => document.getElementById('search-btn')?.click(), 100);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery) return;
    setIsSearching(true);
    setHasSearched(true);
    setSearchResults([]);
    setSearchError('');
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, top_k: 5, sources: activeSources }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Ошибка сервера');
      setSearchResults(data.results || []);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : 'Ошибка соединения с сервером');
    } finally {
      setIsSearching(false);
    }
  };

  const handleAsk = async () => {
    if (!askQuery) return;
    setIsAsking(true);
    setAskResult(null);
    setStreamingAnswer('');
    setAskError('');

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: askQuery, top_k: 5, sources: activeSources }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Ошибка сервера');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';
      let sources: SearchResult[] = [];

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        for (const line of text.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const msg = JSON.parse(line.slice(6));

          if (msg.type === 'chunk') {
            accumulated += msg.text;
            setStreamingAnswer(accumulated);
          } else if (msg.type === 'sources') {
            sources = msg.sources;
          } else if (msg.type === 'error') {
            throw new Error(msg.text);
          }
        }
      }

      setAskResult({ answer: accumulated, sources });
      setStreamingAnswer('');
    } catch (error) {
      setAskError(error instanceof Error ? error.message : 'Ошибка соединения с сервером');
    } finally {
      setIsAsking(false);
    }
  };

  const displayedAnswer = streamingAnswer || askResult?.answer;
  const isAnswerNotFound = displayedAnswer?.toLowerCase().includes("не найден") || displayedAnswer?.toLowerCase().includes("извините");

  return (
    <motion.div
      className="page-container"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="page-header">
        <div className="page-title-group">
          <h1>Анализ и поиск</h1>
          <p>Интеллектуальная работа с текстами книг. Получайте ответы, основанные только на проверенных фрагментах.</p>
        </div>
      </div>

      {books.length === 0 ? (
        <div className="card" style={{textAlign: 'center', padding: '60px 20px'}}>
           <Info size={48} strokeWidth={1.5} style={{opacity: 0.3, marginBottom: 20}} />
           <h2 className="serif">Библиотека пуста</h2>
           <p style={{marginBottom: 24}}>Для начала работы необходимо загрузить хотя бы одну книгу в формате .txt</p>
           <Link to="/" className="primary-btn">
             Перейти в Библиотеку <ArrowRight size={18} />
           </Link>
        </div>
      ) : (
        <div className="search-layout">
          <motion.div 
            className="book-filter"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <span className="book-filter-label"><Filter size={14} style={{verticalAlign: 'middle', marginRight: 6}} /> Область поиска:</span>
            <div className="book-filter-chips">
              <button
                className={`book-chip ${selectedBooks.length === 0 ? 'active' : ''}`}
                onClick={() => setSelectedBooks([])}
              >
                Все книги ({books.length})
              </button>
              {books.map((book) => (
                <button
                  key={book}
                  className={`book-chip ${selectedBooks.includes(book) ? 'active' : ''}`}
                  onClick={() => toggleBook(book)}
                >
                  {book.replace(/\.[^.]+$/, '')}
                </button>
              ))}
            </div>
          </motion.div>

          <div className="tabs-container">
            <button
              className={`tab-btn ${activeTab === 'qa' ? 'active' : ''}`}
              onClick={() => setActiveTab('qa')}
            >
              <Sparkles size={18} />
              AI Ответы
            </button>
            <button
              className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              <SearchIcon size={18} />
              Поиск фрагментов
            </button>
          </div>

          <div className="card search-card">
            {!hasSearched && !isAsking && (
              <div className="quick-examples">
                <span className="book-filter-label">Попробуйте спросить:</span>
                <div className="example-grid">
                  {exampleQueries.map((q, i) => (
                    <button key={i} className="example-item" onClick={() => handleQuickQuery(q.text, q.type as any)}>
                      <span>{q.text}</span>
                      <ExternalLink size={14} />
                    </button>
                  ))}
                </div>
              </div>
            )}

            <AnimatePresence mode="wait">
              {activeTab === 'qa' ? (
                <motion.div 
                  key="qa"
                  initial={{ opacity: 0, x: -10 }} 
                  animate={{ opacity: 1, x: 0 }} 
                  exit={{ opacity: 0, x: 10 }}
                  className="tab-content"
                >
                  <h2 className="serif">Задать вопрос книге</h2>
                  <p className="card-desc">Ответ будет сформирован строго на основе найденных цитат.</p>

                  <div className="search-bar">
                    <input
                      type="text"
                      autoFocus 
                      placeholder="Ваш вопрос..."
                      value={askQuery}
                      onChange={(e) => setAskQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                    />
                    <button id="ask-btn" className="primary-btn" onClick={handleAsk} disabled={isAsking}>
                      {isAsking ? 'Думаю...' : 'Спросить'}
                    </button>
                  </div>

                  {askError && (
                    <motion.div className="status-message error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{marginTop: 20}}>
                      {askError}
                    </motion.div>
                  )}

                  {displayedAnswer && (
                    <motion.div
                      className="ask-result"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <div className={`answer-box ${isAnswerNotFound ? 'not-found' : ''}`}>
                        <div className="answer-header">
                          <span className="ai-badge">AI</span>
                          <h3 className="serif">Ответ системы</h3>
                          {isAsking && <span className="streaming-indicator" />}
                        </div>
                        <p>{displayedAnswer}</p>
                      </div>

                      <AnimatePresence>
                        {askResult?.sources && askResult.sources.length > 0 && (
                          <motion.div 
                            className="sources-section"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                          >
                            <h4><BookOpen size={16} /> Использованные источники ({askResult.sources.length})</h4>
                            <div className="sources-list">
                              {askResult.sources.map((src, i) => (
                                <motion.div 
                                  key={i} 
                                  className="source-card"
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ delay: 0.4 + i * 0.1 }}
                                >
                                  <div className="source-title">
                                    <strong>{src.source}</strong>
                                    <div className="source-details">
                                        {src.tome && <span className="structure-tag">{src.tome}</span>}
                                        {src.chapter && <span className="structure-tag">{src.chapter}</span>}
                                        {src.location && (
                                        <span className="source-location">
                                            <MapPin size={12} />
                                            {src.location}
                                        </span>
                                        )}
                                    </div>
                                  </div>
                                  <div className="source-text">«{src.content}»</div>
                                </motion.div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  )}
                </motion.div>
              ) : (
                <motion.div 
                  key="search"
                  initial={{ opacity: 0, x: 10 }} 
                  animate={{ opacity: 1, x: 0 }} 
                  exit={{ opacity: 0, x: -10 }}
                  className="tab-content"
                >
                  <h2 className="serif">Поиск точных фрагментов</h2>
                  <p className="card-desc">Поможет найти конкретное место в тексте по описанию события или ключевым словам.</p>

                  <div className="search-bar">
                    <input
                      type="text"
                      placeholder="Найди, где говорится про..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    />
                    <button id="search-btn" className="primary-btn" onClick={handleSearch} disabled={isSearching}>
                      {isSearching ? 'Ищу...' : 'Найти'}
                    </button>
                  </div>

                  {searchError && (
                    <motion.div className="status-message error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{marginTop: 20}}>
                      {searchError}
                    </motion.div>
                  )}

                  <div className="search-results">
                    {searchResults.length === 0 && !isSearching && hasSearched && !searchError && (
                      <div className="card" style={{background: 'var(--error-bg)', color: 'var(--error-text)', textAlign: 'center'}}>
                        <Info size={24} style={{marginBottom: 10}} />
                        <p>В загруженных текстах нужного фрагмента не обнаружено.</p>
                      </div>
                    )}
                    {searchResults.map((res, i) => (
                      <motion.div
                        key={i}
                        className="result-item"
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.08 }}
                      >
                        <div className="result-header">
                          <div className="result-meta">
                            <span className="source-badge">
                              <BookOpen size={16} /> {res.source}
                            </span>
                            <div className="source-details" style={{marginTop: 6}}>
                              {res.tome && <span className="structure-tag">{res.tome}</span>}
                              {res.part && <span className="structure-tag">{res.part}</span>}
                              {res.chapter && <span className="structure-tag">{res.chapter}</span>}
                            </div>
                          </div>
                          <span className="score-badge">Совпадение: {Math.round(res.score * 100)}%</span>
                        </div>
                        <p className="result-text-content">
                          «{res.content}»
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default Search;
