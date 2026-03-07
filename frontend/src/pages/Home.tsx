import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle, AlertCircle, BookOpen, Info, Trash2, XCircle } from 'lucide-react';
import './Home.css';

const API_URL = import.meta.env.VITE_API_URL;

const Home = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'already_indexed'>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [indexProgress, setIndexProgress] = useState(0);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [uploadPhase, setUploadPhase] = useState<'idle' | 'upload' | 'index'>('idle');
  const [books, setBooks] = useState<string[]>([]);
  const [deletingBook, setDeletingBook] = useState<string | null>(null);
  const [indexingProgress, setIndexingProgress] = useState<Record<string, { percent: number; current: number; total: number }>>({});

  const abortControllerRef = useRef<AbortController | null>(null);
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchBooks = async () => {
    try {
      const res = await fetch(`${API_URL}/api/books`);
      const data = await res.json();
      setBooks(data.books || []);
    } catch {
      // Игнорируем ошибки при получении списка книг
    }
  };

  const fetchIndexingProgress = async () => {
    try {
      const res = await fetch(`${API_URL}/api/indexing-progress`);
      const data = await res.json();
      const progress = data.progress || {};
      setIndexingProgress(progress);
      if (Object.keys(progress).length === 0) {
        if (progressPollRef.current) {
          clearInterval(progressPollRef.current);
          progressPollRef.current = null;
        }
        fetchBooks();
      }
    } catch {
      // Игнорируем ошибки при получении прогресса индексации
    }
  };

  const startProgressPolling = () => {
    if (progressPollRef.current) return;
    progressPollRef.current = setInterval(fetchIndexingProgress, 1000);
  };

  const handleDeleteBook = async (filename: string) => {
    setDeletingBook(filename);
    try {
      const res = await fetch(`${API_URL}/api/book/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (res.ok) fetchBooks();
    } finally {
      setDeletingBook(null);
    }
  };

  useEffect(() => {
    fetchBooks();
    fetchIndexingProgress().then(() => startProgressPolling());
    return () => {
      if (progressPollRef.current) clearInterval(progressPollRef.current);
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUploadStatus('idle');
      setStatusMessage('');
    }
  };

  const handleCancelUpload = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setUploadStatus('idle');
      setUploadPhase('idle');
      setStatusMessage('Загрузка отменена');
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploadStatus('loading');
    setUploadProgress(0);
    setIndexProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
    setUploadPhase('upload');
    setStatusMessage('');

    const formData = new FormData();
    formData.append('file', file);

    abortControllerRef.current = new AbortController();

    try {
      // Так как fetch не поддерживает прогресс загрузки напрямую,
      // мы просто переключаем фазу. Для текстовых файлов это происходит быстро
      setUploadProgress(100); 

      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        let detail = 'Ошибка при загрузке';
        try {
          const err = await response.json();
          detail = err.detail || detail;
        } catch {
          // Игнорируем ошибки разбора JSON
        }
        setUploadStatus('error');
        setStatusMessage(detail);
        setUploadPhase('idle');
        return;
      }

      setUploadPhase('index');
      startProgressPolling();
      const reader = response.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.type === 'start') {
                setTotalChunks(data.total_chunks);
              } else if (data.type === 'progress') {
                setIndexProgress(data.percent);
                setCurrentChunk(data.current);
                setTotalChunks(data.total);
              } else if (data.type === 'already_indexed') {
                setUploadStatus('already_indexed');
                setStatusMessage(data.message || 'Книга уже проиндексирована');
                setUploadPhase('idle');
                return;
              } else if (data.type === 'success') {
                setUploadStatus('success');
                setStatusMessage(`Книга успешно обработана! Добавлено ${data.chunks_added} фрагментов.`);
                fetchBooks();
                setFile(null);
                setUploadPhase('idle');
                return;
              } else if (data.type === 'error') {
                setUploadStatus('error');
                setStatusMessage(data.detail || 'Ошибка при индексации');
                setUploadPhase('idle');
                return;
              }
            } catch (e) {
              console.error('Ошибка парсинга SSE:', e);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setUploadStatus('error');
      setStatusMessage('Ошибка соединения с сервером');
      setUploadPhase('idle');
    }
  };

  return (
    <motion.div
      className="page-container"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="page-header">
        <div className="page-title-group">
          <h1>Управление книгами</h1>
          <p>Загружайте тексты в формате .txt для создания базы знаний. Система автоматически разобьет их на фрагменты для умного поиска.</p>
        </div>
        <div className="page-note">TXT, UTF-8</div>
      </div>

      <div className="home-grid">
        <div className="card upload-card">
          <div className={`upload-area ${file ? 'has-file' : ''}`}>
            <input
              type="file"
              id="file-upload"
              accept=".txt"
              onChange={handleFileChange}
              className="hidden-input"
              disabled={uploadStatus === 'loading'}
            />
            <label htmlFor="file-upload" className="upload-label">
              <UploadCloud size={48} className="upload-icon" />
              <span className="upload-text">{file ? file.name : 'Выберите файл книги (.txt)'}</span>
              <span className="upload-hint">Перетащите сюда файл или кликните для выбора</span>
              {file && (
                <motion.span 
                  className="file-size"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  {(file.size / 1024 / 1024).toFixed(1)} МБ
                </motion.span>
              )}
            </label>
          </div>

          <AnimatePresence>
            {file && (
              <motion.div 
                className="upload-actions"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                {uploadStatus === 'loading' ? (
                  <button
                    className="secondary-btn cancel-btn"
                    onClick={handleCancelUpload}
                  >
                    <XCircle size={18} style={{ marginRight: 8 }} />
                    Отменить загрузку
                  </button>
                ) : (
                  <button
                    className="primary-btn"
                    onClick={handleUpload}
                    disabled={false}
                  >
                    Загрузить и проиндексировать
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {uploadStatus === 'loading' && (
              <motion.div 
                className="upload-progress-wrap"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <div className="upload-progress-bar">
                  <motion.div
                    className="upload-progress-fill"
                    initial={{ width: 0 }}
                    animate={{ 
                      width: uploadPhase === 'upload' ? `${uploadProgress}%` : `${indexProgress}%`,
                      backgroundColor: '#000000'
                    }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <div className="upload-progress-label">
                  {uploadPhase === 'upload' ? (
                    <>
                      <span>Передача файла...</span>
                      <span>{uploadProgress}%</span>
                    </>
                  ) : (
                    <>
                      <span>Индексация: {currentChunk} из {totalChunks} чанков</span>
                      <span>{indexProgress}%</span>
                    </>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {statusMessage && (
            <motion.div
              className={`status-message ${uploadStatus}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
            >
              {uploadStatus === 'success' && <CheckCircle size={18} />}
              {uploadStatus === 'error' && <AlertCircle size={18} />}
              {uploadStatus === 'already_indexed' && <Info size={18} />}
              <span>{statusMessage}</span>
            </motion.div>
          )}
        </div>

        <div className="card library-card">
          <div className="library-meta">
            <h2 className="serif">База знаний</h2>
            <motion.span 
              key={books.length + Object.keys(indexingProgress).length}
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
            >
              {books.length + Object.keys(indexingProgress).filter(f => !books.includes(f)).length} книги
            </motion.span>
          </div>

          {books.length === 0 && Object.keys(indexingProgress).length === 0 ? (
            <div className="empty-state">
              <BookOpen size={48} strokeWidth={1} style={{ opacity: 0.2, marginBottom: 16 }} />
              <p>Система готова к работе. Загрузите файлы для поиска.</p>
            </div>
          ) : (
            <ul className="books-list">
              <AnimatePresence mode="popLayout">
                {Object.entries(indexingProgress).map(([filename, prog]) => (
                  <motion.li
                    key={`indexing-${filename}`}
                    className="book-item"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    layout
                  >
                    <div style={{ background: '#fff8e1', color: '#b45309', padding: 10, borderRadius: 12 }}>
                      <BookOpen size={20} />
                    </div>
                    <div className="book-main" style={{ flex: 1 }}>
                      <span className="book-name">{filename}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                        <div style={{ flex: 1, height: 4, background: '#e5e7eb', borderRadius: 2, overflow: 'hidden' }}>
                          <motion.div
                            style={{ height: '100%', background: '#b45309', borderRadius: 2 }}
                            animate={{ width: `${prog.percent}%` }}
                            transition={{ duration: 0.3 }}
                          />
                        </div>
                        <span style={{ fontSize: 12, color: '#b45309', fontWeight: 600, whiteSpace: 'nowrap' }}>
                          {prog.current} / {prog.total} чанков ({prog.percent}%)
                        </span>
                      </div>
                    </div>
                  </motion.li>
                ))}
                {books.filter(book => !indexingProgress[book]).map((book, i) => (
                  <motion.li
                    key={book}
                    className="book-item"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ delay: i * 0.05 }}
                    layout
                  >
                    <div style={{ background: '#eef7f1', color: '#1e4620', padding: 10, borderRadius: 12 }}>
                      <CheckCircle size={20} className="icon" />
                    </div>
                    <div className="book-main">
                      <span className="book-name">{book}</span>
                      <span className="book-meta" style={{color: '#1e4620', fontWeight: 600}}>Проиндексировано и готово</span>
                    </div>
                    <button
                      className="delete-btn"
                      onClick={() => handleDeleteBook(book)}
                      disabled={deletingBook === book}
                      title="Удалить из системы"
                    >
                      <Trash2 size={18} />
                    </button>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Home;
