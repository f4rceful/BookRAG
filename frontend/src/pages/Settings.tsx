import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Save, RefreshCw, Cpu, Terminal, HelpCircle } from 'lucide-react';
import './Settings.css';

const Settings = () => {
  const [currentModel, setCurrentModel] = useState<string>('Загрузка...');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [status, setStatus] = useState<{type: 'idle' | 'loading' | 'success' | 'error', message: string}>({type: 'idle', message: ''});

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      // Получение текущей модели
      const resModels = await fetch(`${import.meta.env.VITE_API_URL}/api/models`);
      const dataModels = await resModels.json();
      setCurrentModel(dataModels.current_model);
      setSelectedModel(dataModels.current_model);

      // Получение доступных моделей из Ollama
      const resAvail = await fetch(`${import.meta.env.VITE_API_URL}/api/models/available`);
      const dataAvail = await resAvail.json();
      setAvailableModels(dataAvail.models || []);
    } catch (error) {
      console.error('Ошибка при получении данных', error);
      setCurrentModel('Ошибка соединения');
    }
  };

  const handleSetModel = async () => {
    if (!selectedModel) return;
    try {
      setStatus({ type: 'loading', message: 'Переключение модели...' });
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel }),
      });
      const data = await response.json();
      if (response.ok) {
        setCurrentModel(data.model);
        setStatus({ type: 'success', message: 'Модель успешно изменена!' });
      } else {
        setStatus({ type: 'error', message: `Ошибка: ${data.detail}` });
      }
    } catch (error) {
      setStatus({ type: 'error', message: 'Ошибка сети.' });
    }
    
    setTimeout(() => {
      setStatus(prev => prev.type === 'success' ? { type: 'idle', message: '' } : prev);
    }, 3000);
  };

  return (
    <motion.div 
      className="page-container"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="page-header">
        <div className="page-title-group">
          <h1>Конфигурация</h1>
          <p>Настройка параметров искусственного интеллекта и выбор локальных языковых моделей.</p>
        </div>
        <button className="refresh-btn" onClick={fetchData} title="Обновить список моделей">
          <RefreshCw size={20} />
        </button>
      </div>

      <div className="card settings-card">
        <div style={{display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8}}>
            <Cpu size={24} strokeWidth={2} />
            <h2 className="serif">Модель генерации</h2>
        </div>
        <p className="card-desc">
          Выберите LLM (Large Language Model), которая будет отвечать за анализ текста и формирование ответов. Рекомендуется использовать модели объемом от 7B параметров для лучшего качества.
        </p>
        
        <div className="current-model-info">
          <span>Активная модель</span>
          <span className="model-badge">{currentModel}</span>
        </div>

        <div className="settings-form">
          <div className="input-group">
            <label>Доступные в системе:</label>
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={status.type === 'loading'}
            >
              <option value="" disabled>-- Выберите модель из списка --</option>
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
              {availableModels.length === 0 && <option value="" disabled>Модели не найдены</option>}
            </select>
          </div>
          <button 
            className="primary-btn save-btn"
            onClick={handleSetModel} 
            disabled={status.type === 'loading' || !selectedModel || selectedModel === currentModel}
          >
            <Save size={20} />
            {status.type === 'loading' ? 'Применяю...' : 'Применить'}
          </button>
        </div>

        <AnimatePresence>
          {status.message && (
            <motion.div 
              className={`settings-status ${status.type}`}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              {status.message}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="recommendations">
          <div style={{display: 'flex', alignItems: 'center', gap: 10}}>
            <HelpCircle size={20} color="var(--text-secondary)" />
            <h3 className="serif">Как расширить список?</h3>
          </div>
          <p>Для добавления новых моделей используйте CLI Ollama. Мы рекомендуем следующие модели:</p>
          <div style={{display: 'flex', flexDirection: 'column', gap: 12}}>
            <div>
                <span style={{fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-muted)'}}>ОПТИМАЛЬНО ДЛЯ РУССКОГО:</span>
                <code>ollama pull gemma3:12b</code>
            </div>
            <div>
                <span style={{fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-muted)'}}>ЛЕГКАЯ И БЫСТРАЯ:</span>
                <code>ollama pull llama3.2:3b</code>
            </div>
          </div>
          <p style={{marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: 6}}>
            <Terminal size={14} /> После загрузки нажмите кнопку обновления вверху страницы.
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default Settings;
