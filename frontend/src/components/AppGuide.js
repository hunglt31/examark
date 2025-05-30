import React from 'react';
import './AppGuide.css';

function AppGuide() {
  return (
    <div className="AppGuide">
      <h2>How to Use the App</h2>
      <ol>
        <li>
          <strong>Navigation:</strong> Use the menu buttons at the top to access sections such as Grade Exam, View Results, View Summary and Help.
        </li>
        <li>
          <strong>Grade Exams:</strong> Click on "Grade Exam" to start reviewing and grading exams. Follow the on-screen instructions to upload and submit your grading data.
        </li>
        <li>
          <strong>View Results:</strong> Check exam statistics and detailed reports by clicking "View Results."
        </li>

        <li>
          <strong>View Summary:</strong>  Access the "View Summary" section to see a summary of all graded exams, including average scores and distribution.
        </li>
        <li>
          <strong>Help:</strong> Need assistance? Visit the Help section to view FAQs or contact support.
        </li>
      </ol>
      <p>
        For additional assistance, please email us at{" "}
        <a href="mailto:Hung.LT216834@sis.hust.edu.vn">Hung.LT216834@sis.hust.edu.vn</a>.
      </p>
    </div>
  );
}

export default AppGuide;