import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './GradeExamPage.css';

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
      
      setGradingMessage(`Grading job started with ID: ${newJobId}. Processing...`);
      
      // Start polling for status
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
      
      statusCheckInterval.current = setInterval(() => {
        checkGradingStatus(newJobId);
      }, 30000); 
      
    } catch (error) {
      console.error("Error sending grading request:", error);
      setGradingMessage("An error occurred while communicating with the server. Please try again.");
      setIsGrading(false);
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
      
      if (statusData.status === "completed") {
        clearInterval(statusCheckInterval.current);
        setGradingMessage("Grading completed! Fetching results...");
        setIsGradingComplete(true);
        
        // Fetch results
        fetchResults(id);
      }
    } catch (error) {
      console.error("Error checking status:", error);
      setGradingMessage(`Error checking grading status: ${error.message}`);
      setIsGrading(false);
    }
  };
  
  // Fetch results once grading is complete
 const fetchResults = async (id) => {
  try {
    // Fetch CSV data
    const csvResponse = await fetch(`http://localhost:8080/results/${id}/csv`);
    if (!csvResponse.ok) {
      throw new Error(`Failed to fetch CSV: ${csvResponse.status}`);
    }
    
    const csvText = await csvResponse.text();
    setCsvData(csvText);
    
    // Fetch image list
    const imagesResponse = await fetch(`http://localhost:8080/results/${id}/images`);
    if (!imagesResponse.ok) {
      throw new Error(`Failed to fetch image list: ${imagesResponse.status}`);
    }
    
    const imagesData = await imagesResponse.json();
    setImages(imagesData.images || []);
    
    // Clear any previous edits
    localStorage.removeItem('examarkEdits');
    
    // Save results to localStorage for other pages
    localStorage.setItem('examarkJobId', id);
    localStorage.setItem('examarkCsvData', csvText);
    localStorage.setItem('examarkImages', JSON.stringify(imagesData.images || []));
    
    setGradingMessage("Grading request completed successfully!");
    setIsGrading(false);
    setShowNavigationOptions(true);
    
  } catch (error) {
    console.error("Error fetching results:", error);
    setGradingMessage(`Error fetching results: ${error.message}`);
    setIsGrading(false);
  }
};
  
// Navigation handlers
const navigateToResults = () => {
  navigate('/results?refresh=' + new Date().getTime());
};

const navigateToSummarize = () => {
  navigate('/summarize?refresh=' + new Date().getTime());
};

  return (
    <div className="ExamPage">
      <header className="grade-header">
        <div className="grade-header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="grade-header-logo" />
        </div>
        <div className="grade-header-center">
          <h1>Grade Exam</h1>
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
              <p>Upload the exam paper (PDF) and the answer key (CSV) to begin grading.</p>

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
                  View Results Page
                  <span className="description">Detailed view of each page with answers</span>
                </button>
                
                <button 
                  onClick={navigateToSummarize} 
                  className="nav-button summarize-button"
                >
                  <i className="fas fa-table"></i>
                  View CSV Summary
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
                }}
              >
                Grade Another Exam
              </button>
            </div>
          )}
          
          {gradingMessage && <p className="grading-message">{gradingMessage}</p>}
          
          <Link to="/">
            <button className="btn btn-secondary btn-medium">Back to Main Page</button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default GradeExamPage;