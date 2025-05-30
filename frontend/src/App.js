import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import MainPage from './pages/MainPage';
import GradeExamPage from './pages/GradeExamPage';
import ResultsPage from './pages/ResultsPage';
import SummarizePage from './pages/SummarizePage';
import HelpPage from './pages/HelpPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/grade" element={<GradeExamPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/summarize" element={<SummarizePage />} />
        <Route path="/help" element={<HelpPage />} />
      </Routes>
    </Router>
  );
}

export default App;