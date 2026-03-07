import { Link, useLocation } from 'react-router-dom';
import { BookOpen, Search, Settings } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <BookOpen className="icon" size={28} />
          <span>BookRAG</span>
        </Link>
        <div className="navbar-links">
          <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
            Библиотека
          </Link>
          <Link to="/search" className={location.pathname === '/search' ? 'active' : ''}>
            <Search size={18} className="nav-icon" />
            Поиск
          </Link>
          <Link to="/settings" className={location.pathname === '/settings' ? 'active' : ''}>
            <Settings size={18} className="nav-icon" />
            Настройки
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
