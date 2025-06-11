import React from 'react';
import './AppGuide.css';

function AppGuide() {
  return (
    <div className="AppGuide">
      <h2>How to Use the App</h2>
      <ol>
        <li>
          <strong>Navigation:</strong> Use the menu buttons at the top to access sections such as Start Auto Grading, Review Exam Details, Review Results Sheet and Help.
        </li>
        <li>
          <strong>Start Auto Grading:</strong> Click on "Start Auto Grading" to start reviewing and grading exams. Follow the on-screen instructions to upload and submit your grading data.
        </li>
        <li>
          <strong>Review Exam Details:</strong> Check exam statistics and detailed reports by clicking "Review Exam Details."
        </li>

        <li>
          <strong>Review Results Sheet:</strong>  Access the "Review Results Sheet" section to see a spreadsheet of all graded exams, including answers and scores in details.
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