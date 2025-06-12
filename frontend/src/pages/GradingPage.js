import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './GradingPage.css';
import CustomAlert from '../components/CustomAlert';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

function GradeExamPage() {
  const [gradingMessage, setGradingMessage] = useState('');
  const [pdfFile, setPdfFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  
  // States for results
  const [jobId, setJobId] = useState(null);
  const [isGrading, setIsGrading] = useState(false);
  const [isGradingComplete, setIsGradingComplete] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [showNavigationOptions, setShowNavigationOptions] = useState(false);
  
  // Progress tracking states
  const [progress, setProgress] = useState({
    stage: '',
    step: '',
    currentPage: 0,
    totalPages: 0,
    progressPercent: 0.0
  });

  // Refs
  const pdfInputRef = useRef(null);
  const csvInputRef = useRef(null);
  const statusCheckInterval = useRef(null);
  const navigate = useNavigate();

  // Clean up interval on component unmount
  useEffect(() => {
    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
    };
  }, []);

  // Sample CSV download functionality
  const downloadSampleCSV = () => {
    const sampleCSVContent = `,ExamID,101,102
Part,Question,Key,Key
1,1,A,A
1,2,B,D
1,3,A,B
1,4,A,C
1,5,A,B
1,6,B,C
1,7,A,A
1,8,C,C
1,9,C,C
1,10,A,A
1,11,C,C
1,12,A,A
1,13,C,C
1,14,A,C
1,15,B,C
1,16,B,B
2,1,AB,ACD
2,2,DF,BCD
2,3,AC,BCDE
2,4,ACD,BCDF
2,5,ACDE,ACD
2,6,BCDE,ACDEF
2,7,CEF,CEF
2,8,ACD,ACDEF`;

    const blob = new Blob([sampleCSVContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'sample_answer_key.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  // Handle file input changes
  const handlePdfFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setPdfFile(file);
      setGradingMessage(csvFile ? `PDF: ${file.name}, CSV: ${csvFile.name}` : `PDF selected: ${file.name}`);
    } else {
      setPdfFile(null);
      setGradingMessage(csvFile ? `CSV selected: ${csvFile.name}` : '');
    }
  };

  const handleCsvFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setCsvFile(file);
      setGradingMessage(pdfFile ? `PDF: ${pdfFile.name}, CSV: ${file.name}` : `CSV selected: ${file.name}`);
      
      // Read and save the CSV content to localStorage for future use
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvContent = e.target.result;
        localStorage.setItem('examarkAnswerKey', csvContent);
        localStorage.setItem('examarkAnswerKeyFileName', file.name);
      };
      reader.readAsText(file);
    } else {
      setCsvFile(null);
      setGradingMessage(pdfFile ? `PDF selected: ${pdfFile.name}` : '');
    }
  };

  // Start grading process
  const handleGradeExam = async () => {
    if (!pdfFile || !csvFile) {
      setGradingMessage("Please upload both the exam PDF and the answer CSV file.");
      return;
    }

    setGradingMessage(`Uploading and initiating grading for PDF: ${pdfFile.name} with answers from: ${csvFile.name}...`);
    setIsGrading(true);
    setIsGradingComplete(false);
    setCsvData(null);
    setImages([]);
    setShowNavigationOptions(false);
    setProgress({
      stage: 'initializing',
      step: 'Starting upload...',
      currentPage: 0,
      totalPages: 0,
      progressPercent: 0.0
    });

    const formData = new FormData();
    formData.append('pdfFile', pdfFile);
    formData.append('csvFile', csvFile);

    try {
      const response = await fetch('http://localhost:8080/grade', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      const newJobId = result.jobId;
      setJobId(newJobId);
      localStorage.setItem('examarkAnswerKeyJobId', newJobId);
      setGradingMessage(`Grading job started with ID: ${newJobId}. Processing...`);
      
      // Start polling for status immediately and more frequently
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
      
      // Check immediately
      checkGradingStatus(newJobId);
      
      // Then check every 2 seconds for better responsiveness
      statusCheckInterval.current = setInterval(() => {
        checkGradingStatus(newJobId);
      }, 2000); 
      
    } catch (error) {
      console.error("Error sending grading request:", error);
      setGradingMessage("An error occurred while communicating with the server. Please try again.");
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to start grading',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0
      });
    }
  };
  
  // Check the status of grading
  const checkGradingStatus = async (id) => {
    try {
      const response = await fetch(`http://localhost:8080/status/${id}`);
      
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
      }
      
      const statusData = await response.json();
      
      // Update progress state
      setProgress({
        stage: statusData.currentStage || '',
        step: statusData.currentStep || '',
        currentPage: statusData.currentPage || 0,
        totalPages: statusData.totalPages || 0,
        progressPercent: statusData.progress || 0.0
      });
      
      // Update grading message with current progress
      let progressMessage = statusData.currentStep || 'Processing...';
      if (statusData.totalPages > 0) {
        progressMessage += ` (${statusData.currentPage}/${statusData.totalPages})`;
      }
      if (statusData.progress > 0) {
        progressMessage += ` - ${Math.round(statusData.progress)}%`;
      }
      setGradingMessage(progressMessage);
      
      if (statusData.status === "completed") {
        clearInterval(statusCheckInterval.current);
        setGradingMessage("Grading completed! Fetching results...");
        setIsGradingComplete(true);
        
        // Fetch results
        fetchResults(id);
      } else if (statusData.status === "error") {
        clearInterval(statusCheckInterval.current);
        setGradingMessage(`Error: ${statusData.error || 'Grading failed'}`);
        setIsGrading(false);
        setProgress({
          stage: 'error',
          step: statusData.error || 'Grading failed',
          currentPage: 0,
          totalPages: 0,
          progressPercent: 0.0
        });
      }
    } catch (error) {
      console.error("Error checking status:", error);
      setGradingMessage(`Error checking grading status: ${error.message}`);
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to check status',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0
      });
    }
  };

  // [MinIO] Fetch results once grading is complete
  const fetchResults = async (id) => {
    try {
      // Fetch CSV data from MinIO via backend
      const csvResponse = await fetch(`http://localhost:8080/results/${id}/csv`);
      if (!csvResponse.ok) {
        throw new Error(`Failed to fetch CSV: ${csvResponse.status}`);
      }
      
      const csvText = await csvResponse.text();
      setCsvData(csvText);
      
      // Fetch image list from MinIO via backend - returns MinIO URLs
      const imagesResponse = await fetch(`http://localhost:8080/results/${id}/images`);
      if (!imagesResponse.ok) {
        throw new Error(`Failed to fetch image list: ${imagesResponse.status}`);
      }
      
      const imagesData = await imagesResponse.json();
      
      // Images now come with MinIO URLs directly
      const imageUrls = imagesData.images.map(img => ({
        name: img.name,
        url: img.url  // Direct MinIO URL
      }));
      
      setImages(imageUrls);
      
      // Clear any previous edits
      localStorage.removeItem('examarkEdits');
      
      // Save results to localStorage for other pages
      localStorage.setItem('examarkJobId', id);
      localStorage.setItem('examarkCsvData', csvText);
      localStorage.setItem('examarkImages', JSON.stringify(imageUrls));
      
      setGradingMessage("Grading request completed successfully!");
      setIsGrading(false);
      setShowNavigationOptions(true);
      setProgress({
        stage: 'completed',
        step: 'All done!',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 100.0
      });
      
    } catch (error) {
      console.error("Error fetching results:", error);
      setGradingMessage(`Error fetching results: ${error.message}`);
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to fetch results',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0
      });
    }
  };
  
  // Navigation handlers
  const navigateToResults = () => {
    navigate('/results?refresh=' + new Date().getTime());
  };

  const navigateToSheet = () => {
    navigate('/sheet?refresh=' + new Date().getTime());
  };

  // Progress bar component
  const ProgressBar = () => {
    if (!isGrading && !isGradingComplete) return null;
    
    return (
      <div className="progress-container">
        <div className="progress-info">
          <div className="progress-stage">
            <strong>Stage:</strong> {progress.stage.replace('_', ' ').toUpperCase()}
          </div>
          <div className="progress-step">
            {progress.step}
          </div>
          {progress.totalPages > 0 && (
            <div className="progress-pages">
              Page {progress.currentPage} of {progress.totalPages}
            </div>
          )}
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${Math.max(0, Math.min(100, progress.progressPercent))}%` }}
          ></div>
        </div>
        <div className="progress-percentage">
          {Math.round(progress.progressPercent)}%
        </div>
      </div>
    );
  };

  return (
    <div className="ExamPage">
      <header className="grade-header">
        <div className="grade-header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="grade-header-logo" />
        </div>
        <div className="grade-header-center">
          <h1>Grade Exams</h1>
          <p>Upload your exam files to begin automated grading</p>
        </div>
        <div className="grade-header-right">
          <img src={FamiLogo} alt="Fami Logo" className="grade-header-fami-logo" />
        </div>
      </header>
      
      <div className="exam-container">
        <div className="exam-content">
          {!showNavigationOptions ? (
            <>
              <p>Upload the exam papers (PDF) and the answer key (CSV) to begin grading.</p>

              <div className="file-upload-section">
                <div className="file-upload-area">
                  <button
                    type="button"
                    onClick={() => pdfInputRef.current && pdfInputRef.current.click()}
                    className="btn btn-primary btn-large"
                    disabled={isGrading}
                  >
                    <i className="fas fa-file-pdf"></i> Upload Exam PDF
                  </button>
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handlePdfFileChange}
                    ref={pdfInputRef}
                    style={{ display: 'none' }}
                    id="pdf-upload"
                    disabled={isGrading}
                  />
                  {pdfFile && <span className="file-name">PDF: {pdfFile.name}</span>}
                </div>

                <div className="file-upload-area">
                  <button
                    type="button"
                    onClick={() => csvInputRef.current && csvInputRef.current.click()}
                    className="btn btn-info btn-large"
                    disabled={isGrading}
                  >
                    <i className="fas fa-file-csv"></i> Upload Answer CSV
                  </button>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleCsvFileChange}
                    ref={csvInputRef}
                    style={{ display: 'none' }}
                    id="csv-upload"
                    disabled={isGrading}
                  />
                  {csvFile && <span className="file-name">CSV: {csvFile.name}</span>}
                </div>
              </div>

              <button 
                onClick={handleGradeExam} 
                className="btn btn-success btn-xl" 
                disabled={!pdfFile || !csvFile || isGrading}
              >
                {isGrading ? "Grading in progress..." : "Start Grading"}
              </button>
              
              <ProgressBar />

              {gradingMessage && <p className="grading-message">{gradingMessage}</p>}
              
              {/* CSV Helper Section - hidden when grading is in progress */}
              {!isGrading && !isGradingComplete && (
                <div className="csv-helper-section">
                  <div className="helper-info">
                    <h3><i className="fas fa-question-circle"></i> Need help with the key file format?</h3>
                    <p>Download our sample answer key template to understand the correct format.</p>
                    <button
                      type="button"
                      onClick={downloadSampleCSV}
                      className="btn btn-outline btn-medium"
                    >
                      <i className="fas fa-download"></i> Download Sample Key
                    </button>
                  </div>
                  <div className="csv-format-info">
                    <h4>Key File Format Guidelines:</h4>
                    <ul>
                      <li><strong>Header Row:</strong> Contains ExamID and exam ids (101, 102, etc.)</li>
                      <li><strong>Part Column:</strong> Indicates the section number (1, 2, etc.)</li>
                      <li><strong>Question Column:</strong> Question number within each part</li>
                      <li><strong>Key Columns:</strong> Correct answers for each exam version</li>
                      <li><strong>Multiple Choice:</strong> Use single letters (A, B, C, D)</li>
                      <li><strong>Multiple Selection:</strong> Use combinations (AB, ACD, BCDE)</li>
                    </ul>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="navigation-options">
              <h3>Grading Complete!</h3>
              <p>Your results have been processed successfully. Where would you like to view them?</p>
              
              <div className="navigation-buttons">
                <button 
                  onClick={navigateToResults} 
                  className="nav-button results-button"
                >
                  <i className="fas fa-list-alt"></i>
                  View Grading Results
                  <span className="description">Detailed view of each page with answers</span>
                </button>
                
                <button 
                  onClick={navigateToSheet} 
                  className="nav-button sheet-button"
                >
                  <i className="fas fa-table"></i>
                  View Grading Sheet
                  <span className="description">Edit all exam results in spreadsheet format</span>
                </button>
              </div>
              
              <button
                className="btn btn-secondary btn-medium"
                onClick={() => {
                  setIsGradingComplete(false);
                  setIsGrading(false);
                  setCsvData(null);
                  setImages([]);
                  setPdfFile(null);
                  setCsvFile(null);
                  setShowNavigationOptions(false);
                  setGradingMessage('');
                  setProgress({
                    stage: '',
                    step: '',
                    currentPage: 0,
                    totalPages: 0,
                    progressPercent: 0.0
                  });
                }}
              >
                Grade Another Exam
              </button>
            </div>
          )}
          
          <Link to="/">
            <button className="btn btn-secondary btn-medium">Back to Main Page</button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default GradeExamPage;