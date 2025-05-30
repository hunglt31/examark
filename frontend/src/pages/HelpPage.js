import React from 'react';
import { Link } from 'react-router-dom';

function HelpPage() {
  return (
    <div className="HelpPage">
      <h2>Help & FAQ</h2>
      <p>Here you can provide information on how to use the exam grading system.</p>
      <ul>
        <li>To grade an exam, go to the Grade Exam page and click the button.</li>
        <li>After grading, view the results on the Results page.</li>
        <li>Contact support for any issues.</li>
      </ul>
      <br />
      <Link to="/">
        <button>Back to Dashboard</button>
      </Link>
    </div>
  );
}

export default HelpPage;