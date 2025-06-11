import React from 'react';
import { Link } from 'react-router-dom';
import './HelpPage.css';

function HelpPage() {
  return (
    <div className="HelpPage">
      <div className="help-container">
        <div className="help-header">
          <h1>Examark Help Center</h1>
          <p>Your comprehensive guide to automated exam grading</p>
        </div>
        
        <div className="help-content">
          {/* Overview Section */}
          <div className="help-section">
            <h2>📚 About Examark</h2>
            <p>
              Examark is an intelligent exam grading system designed to automate the process of grading 
              multiple-choice exams. Using advanced YOLO (You Only Look Once) computer vision technology, 
              Examark can accurately detect and grade answer selections in PDF exam papers.
            </p>
            
            <div className="feature-grid">
              <div className="feature-card">
                <div className="feature-icon">🎯</div>
                <h4>YOLO Detection</h4>
                <p>Advanced computer vision model that accurately detects answer selections in real-time</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📊</div>
                <h4>Detailed Results</h4>
                <p>Comprehensive scoring breakdown with individual question analysis and total points calculation</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">✏️</div>
                <h4>Manual Review</h4>
                <p>Review and edit any answers or scores before finalizing results with color-coded feedback</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📈</div>
                <h4>Export Results</h4>
                <p>Export graded results to Excel format for further analysis and record keeping</p>
              </div>
            </div>
          </div>

          {/* How to Use Section */}
          <div className="help-section">
            <h2>🚀 How to Use Examark</h2>
            <div className="workflow-steps">
              <div className="workflow-step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h4>Upload PDF Exam Files</h4>
                  <p>Go to "Start Auto Grading" and upload PDF files of exam papers (only PDF format supported)</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h4>Start Grading Process</h4>
                  <p>Click "Start Grading" to begin automatic YOLO detection and answer recognition</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h4>Review Results</h4>
                  <p>Visit "Review Results Sheet" to view graded papers and verify accuracy with color-coded feedback</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h4>Edit if Needed</h4>
                  <p>Make corrections to any misread answers or student information based on color indicators</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h4>Export Results</h4>
                  <p>Download the final graded results as an Excel file</p>
                </div>
              </div>
            </div>
          </div>

          {/* Color Coding System */}
          <div className="help-section">
            <h2>🎨 Understanding Color Indicators</h2>
            <p>Examark uses a color-coding system to help you quickly identify areas that need attention:</p>
            
            <div className="color-indicators">
              <div className="color-card cyan">
                <div className="color-sample cyan-bg"></div>
                <div className="color-info">
                  <h4>Cyan Background</h4>
                  <p><strong>Model Suggested Answer:</strong> The YOLO model couldn't detect a clear answer selection, so it provides a suggested answer based on common patterns. Please review and confirm or edit as needed.</p>
                </div>
              </div>
              
              <div className="color-card yellow">
                <div className="color-sample yellow-bg"></div>
                <div className="color-info">
                  <h4>Yellow Background</h4>
                  <p><strong>Invalid Format:</strong> Multiple answers were detected for a single question (more than one option selected). Please review the original paper and select the correct single answer.</p>
                </div>
              </div>
              
              <div className="color-card white">
                <div className="color-sample white-bg"></div>
                <div className="color-info">
                  <h4>White Background</h4>
                  <p><strong>Correctly Detected:</strong> The YOLO model successfully detected a clear, single answer selection. No action needed unless you notice an error.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div className="help-section">
            <h2>⚡ Key Features</h2>
            <ul>
              <li>YOLO-powered computer vision for accurate answer detection</li>
              <li>PDF file format support for exam processing</li>
              <li>Automatic detection of student ID and exam information</li>
              <li>Multiple-choice exam structure support (Part 1 & Part 2)</li>
              <li>Real-time grading progress tracking</li>
              <li>Color-coded feedback system for easy review</li>
              <li>Editable results with manual correction capabilities</li>
              <li>Comprehensive scoring breakdown (Part 1, Part 2, Total Points)</li>
              <li>Excel export functionality for grade management</li>
              <li>Responsive design for desktop and mobile use</li>
            </ul>
          </div>

          {/* FAQ Section */}
          <div className="help-section">
            <h2>❓ Frequently Asked Questions</h2>
            
            <div className="faq-item">
              <div className="faq-question">What file formats are supported?</div>
              <div className="faq-answer">
                Examark only supports PDF files. Please ensure your exam papers are saved or scanned as PDF documents before uploading.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">Can Examark grade handwritten responses?</div>
              <div className="faq-answer">
                No, Examark is designed specifically for multiple-choice questions with clearly marked answer selections (bubbles, checkboxes, etc.). It does not process handwritten text or essay responses.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">What PDF quality is required for best results?</div>
              <div className="faq-answer">
                For optimal YOLO detection accuracy, ensure PDFs are clear, well-lit, and high resolution (at least 300 DPI). 
                Avoid blurry, tilted, or poorly scanned documents. Make sure answer bubbles/checkboxes are clearly visible and properly marked.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">What do the different background colors mean?</div>
              <div className="faq-answer">
                <strong>Cyan:</strong> Model suggested answer (no clear detection)<br/>
                <strong>Yellow:</strong> Invalid format (multiple selections detected)<br/>
                <strong>White:</strong> Successfully detected single answer
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">Can I edit the grading results?</div>
              <div className="faq-answer">
                Yes! You can edit student IDs, exam IDs, and individual answers in the Results page. 
                Simply click on any editable field to make changes. Pay special attention to cyan and yellow highlighted fields.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">How accurate is the automatic grading?</div>
              <div className="faq-answer">
                Our YOLO computer vision model achieves high accuracy rates for clear answer selections in well-formatted PDFs. However, 
                we recommend reviewing all results, especially cyan and yellow highlighted areas, before 
                finalizing grades.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">Can I process multiple exams at once?</div>
              <div className="faq-answer">
                Yes! You can upload multiple PDF files in a single batch. The YOLO model will process them 
                sequentially and provide results for all uploaded papers.
              </div>
            </div>

            <div className="faq-item">
              <div className="faq-question">What types of answer formats work best?</div>
              <div className="faq-answer">
                Examark works best with standardized multiple-choice formats like filled bubbles (●), checkboxes (☑), 
                or clearly marked circles. Ensure answer markings are dark and completely fill the designated areas.
              </div>
            </div>
          </div>

          {/* Troubleshooting Section */}
          <div className="help-section">
            <h2>🔧 Troubleshooting</h2>
            <h3>Common Issues and Solutions:</h3>
            <ul>
              <li><strong>Many cyan backgrounds:</strong> PDF quality may be poor or answer markings are faint. Try uploading a higher quality PDF scan</li>
              <li><strong>Yellow backgrounds appearing:</strong> Student may have marked multiple answers. Review the original paper and select the correct single answer</li>
              <li><strong>Wrong student ID detected:</strong> Edit the student ID in the results page manually</li>
              <li><strong>Grading takes too long:</strong> Large PDF files may take time for YOLO processing; be patient</li>
              <li><strong>Missing answers:</strong> Ensure answer bubbles/checkboxes are clearly marked and completely filled</li>
              <li><strong>PDF not uploading:</strong> Check file size and ensure it's a valid PDF format</li>
              <li><strong>Export not working:</strong> Ensure you have results to export and try refreshing the page</li>
            </ul>
          </div>

          {/* Contact Section */}
          <div className="contact-section">
            <h3>Need More Help?</h3>
            <p>If you can't find the answer you're looking for, please contact our support team.</p>
            <div className="contact-info">
              <div className="contact-item">
                <i>📧</i>
                <span>hungthanh3123@gmail.com</span>
              </div>
              <div className="contact-item">
                <i>📞</i>
                <span> (+84) 869 030 103</span>
              </div>
              <div className="contact-item">
                <i>🕒</i>
                <span>Mon-Fri 9AM-5PM EST</span>
              </div>
            </div>
          </div>

          <div className="back-button">
            <Link to="/" className="btn">Back to Dashboard</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HelpPage;