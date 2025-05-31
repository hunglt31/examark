import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import MainPage from './pages/MainPage';
import GradingPage from './pages/GradingPage';
import ResultsPage from './pages/ResultsPage';
import SheetPage from './pages/SheetPage';
import HelpPage from './pages/HelpPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/grade" element={<GradingPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/sheet" element={<SheetPage />} />
        <Route path="/help" element={<HelpPage />} />
      </Routes>
    </Router>
  );
}

export default App;