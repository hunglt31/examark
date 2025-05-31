import React from 'react';
import { Link } from 'react-router-dom';
import './MainPage.css';
import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';
import AppGuide from '../components/AppGuide';

import GradeIcon from '../assets/icons/marking.png';
import ResultsIcon from '../assets/icons/result.png';
import SheetIcon from '../assets/icons/table.png';
import HelpIcon from '../assets/icons/help.png';

function MainPage() {
  return (
    <div className="MainPage">
      {/* Header with university logo, title, and Fami logo */}
      <header className="MainPage-header">
        <div className="header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="header-logo" draggable="false" />
        </div>
        <div className="header-center">
          <h1>EXAMARK DASHBOARD</h1>
        </div>
        <div className="header-right">
          <img src={FamiLogo} alt="Fami Logo" className="header-fami-logo" draggable="false" />
        </div>
      </header>
      
      {/* Navigation Menu */}
      <nav className="MainPage-nav">
        <Link to="/grade">
          <button className="nav-button">
            <img src={GradeIcon} alt="Grade Exam" draggable="false" />
            Grade Exams
          </button>
        </Link>
        <Link to="/results">
          <button className="nav-button">
            <img src={ResultsIcon} alt="Check Results" draggable="false" />
            View Results
          </button>
        </Link>
        <Link to="/sheet">
          <button className="nav-button">
            <img src={SheetIcon} alt="Review Sheet" draggable="false" />
            Review Sheet
          </button>
        </Link>
        <Link to="/help">
          <button className="nav-button">
            <img src={HelpIcon} alt="Help" draggable="false" />
            Help
          </button>
        </Link>
      </nav>

      {/* Unified Dashboard Section */}
      <section className="MainPage-dashboard-grid">
        <div className="dashboard-card">
          <i className="fas fa-hourglass-half card-icon"></i>
          <h3>Pending Exams</h3>
          <p className="dashboard-card-value">5</p>
          <Link to="/grade" className="card-link">Grade Now</Link>
        </div>
        <div className="dashboard-card">
          <i className="fas fa-check-circle card-icon"></i>
          <h3>Graded Today</h3>
          <p className="dashboard-card-value">12</p>
          <Link to="/results" className="card-link">View Results</Link>
        </div>
        <div className="dashboard-card">
          <i className="fas fa-calendar-alt card-icon"></i>
          <h3>Upcoming Exams</h3>
          <p className="dashboard-card-value">3</p>
          <span className="card-link">View Schedule</span> {/* Or Link if you have a schedule page */}
        </div>
        <div className="dashboard-card">
          <i className="fas fa-sync-alt card-icon"></i>
          <h3>System Updates</h3>
          <p className="dashboard-card-value">v1.2</p>
          <span className="card-link">Learn More</span> {/* Or Link to release notes */}
        </div>
        <div className="dashboard-card">
          <i className="fas fa-headset card-icon"></i>
          <h3>Support Tickets</h3>
          <p className="dashboard-card-value">2 Open</p>
          <Link to="/help" className="card-link">Get Support</Link>
        </div>
        {/* Quick Tips card removed */}
      </section>

      {/* App Guide Section */}
      <section className="AppGuide-section">
        <AppGuide />
      </section>

      {/* Footer */}
      <footer className="MainPage-footer">
        <p>© {new Date().getFullYear()} Hanoi University of Science and Technology</p>
        <p>Contact: Hung.LT216834@sis.hust.edu.vn</p>
      </footer>
    </div>
  );
}

export default MainPage;