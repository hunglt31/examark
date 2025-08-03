import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './GradingPage.css';
import CustomAlert from '../components/CustomAlert';
import * as XLSX from 'xlsx';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

// Import sample answer key directly (for testing)
// import sampleAnswerKey from '../../public/answer-keys/correct_answers.xlsx';

function GradeExamPage() {
  const [gradingMessage, setGradingMessage] = useState('');
  const [pdfFile, setPdfFile] = useState(null);
  const [xlsx_file, set_xlsx_file] = useState(null);
  const [valid_files, set_valid_files] = useState(false);

  // States for results
  const [jobId, setJobId] = useState(null);
  const [isGrading, setIsGrading] = useState(false);
  const [isGradingComplete, setIsGradingComplete] = useState(false);
  const [csv_data, set_csv_data] = useState(null);
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [showNavigationOptions, setShowNavigationOptions] = useState(false);

  const TOTAL_NUM_QUESTIONS = 15;

  // Progress tracking states
  const [progress, setProgress] = useState({
    stage: '',
    step: '',
    currentPage: 0,
    totalPages: 0,
    progressPercent: 0.0,
  });

  // Refs
  const pdfInputRef = useRef(null);
  const xlsx_input_ref = useRef(null);
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

  // Function to automatically find and load answer key file
  const autoLoadAnswerKey = async () => {
    try {
      // First, try to get a list of files in the answer-keys folder
      // Since we can't directly list files from frontend, we'll try common patterns
      const possibleFileNames = [
        'correct_answers.xlsx',
        'answer_key.xlsx',
        'key.xlsx',
        'answer.xlsx',
        'exam_key.xlsx',
        'test_key.xlsx',
      ];

      let foundFile = null;
      let fileName = null;

      // Try each possible filename
      for (const name of possibleFileNames) {
        try {
          console.log(`Trying to load: /answer-keys/${name}`);
          const response = await fetch(`/answer-keys/${name}`);
          console.log(`Response status for ${name}:`, response.status, response.statusText);

          if (response.ok) {
            foundFile = await response.arrayBuffer();
            fileName = name;
            console.log(`Successfully loaded file: ${name}, size: ${foundFile.byteLength} bytes`);

            // Debug: Check if file is actually XLSX by looking at first few bytes
            const firstBytes = new Uint8Array(foundFile.slice(0, 8));
            console.log(
              `First 8 bytes of ${name}:`,
              Array.from(firstBytes)
                .map((b) => b.toString(16).padStart(2, '0'))
                .join(' '),
            );

            // XLSX files should start with PK (50 4B) - ZIP file signature
            if (firstBytes[0] === 0x50 && firstBytes[1] === 0x4b) {
              console.log(`${name} appears to be a valid ZIP/XLSX file`);
            } else {
              console.log(`${name} does not appear to be a valid XLSX file (should start with PK)`);
              // Try to read as text to see what it actually contains
              const textDecoder = new TextDecoder();
              const textContent = textDecoder.decode(foundFile.slice(0, 200));
              console.log(`First 200 characters of ${name}:`, textContent);
            }

            break;
          } else {
            console.log(`File not found: ${name} (status: ${response.status})`);
          }
        } catch (error) {
          console.log(`Error loading ${name}:`, error);
          // Continue to next filename
          continue;
        }
      }

      if (!foundFile) {
        throw new Error(
          'No answer key file found in answer-keys folder. Please ensure at least one .xlsx file exists.',
        );
      }

      try {
        console.log('Parsing XLSX file...');
        const workbook = XLSX.read(foundFile, { type: 'array' });
        console.log('Workbook sheets:', workbook.SheetNames);

        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        console.log('Worksheet data range:', worksheet['!ref']);

        const xlsxData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        console.log('Parsed XLSX data:', xlsxData);
        console.log('First row:', xlsxData[0]);
        console.log('Second row:', xlsxData[1]);
        console.log('Third row:', xlsxData[2]);

        // Validate data structure
        if (!xlsxData || xlsxData.length < 3) {
          throw new Error('Invalid file format: File must have at least 3 rows');
        }

        // Convert to your JSON format using existing function
        const simpleJson = create_json_from_array(xlsxData);
        console.log('Auto-loaded answer key JSON:', simpleJson);
        console.log('JSON length:', simpleJson.length);
        if (simpleJson.length > 0) {
          console.log('First exam data:', simpleJson[0]);
        }

        // Create a virtual file object for compatibility
        const virtualFile = {
          name: fileName,
          arrayBuffer: () => Promise.resolve(foundFile),
        };

        set_xlsx_file(virtualFile);
        set_valid_files(true);
        setGradingMessage(`Exam file: ${pdfFile ? pdfFile.name : 'Unknown'}, Auto-loaded answer key: ${fileName}`);

        localStorage.setItem('examarkAnswerKey', JSON.stringify(simpleJson));
        localStorage.setItem('examarkAnswerKeyFileName', fileName);

        return true;
      } catch (parseError) {
        console.error('Error parsing XLSX file:', parseError);
        throw new Error(
          `Failed to parse answer key file: ${parseError.message}. Please ensure the file format is correct.`,
        );
      }
    } catch (error) {
      console.error('Error auto-loading answer key:', error);
      setGradingMessage(
        `Error: ${error.message}. Please ensure at least one .xlsx file exists in the answer-keys folder.`,
      );
      return false;
    }
  };

  // Sample XLSX download functionality
  const download_sample_xlsx = () => {
    const sampleXLSXData = [
      ['ExamID', '101', '102'],
      ['Part', 'Question', 'Key', 'Key'],
      ['1', '1', 'A', 'A'],
      ['1', '2', 'B', 'D'],
      ['1', '3', 'A', 'B'],
      ['1', '4', 'A', 'C'],
      ['1', '5', 'A', 'B'],
      ['1', '6', 'B', 'C'],
      ['1', '7', 'A', 'A'],
      ['1', '8', 'C', 'C'],
      ['1', '9', 'C', 'C'],
      ['1', '10', 'A', 'A'],
      ['2', '1', 'AB', 'ACD'],
      ['2', '2', 'DF', 'BCD'],
      ['2', '3', 'AC', 'BCDE'],
      ['2', '4', 'ACD', 'BCDF'],
      ['2', '5', 'ACDE', 'ACD'],
    ];

    const worksheet = XLSX.utils.aoa_to_sheet(sampleXLSXData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Sample Data');

    const blob = new Blob([XLSX.write(workbook, { bookType: 'xlsx', type: 'array' })], {
      type: 'application/octet-stream',
    });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'sample_answer_key.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  // Function to create and save sample file to answer-keys folder
  const create_sample_in_folder = async () => {
    try {
      const sampleXLSXData = [
        ['ExamID', '101', '102'],
        ['Part', 'Question', 'Key', 'Key'],
        ['1', '1', 'A', 'A'],
        ['1', '2', 'B', 'D'],
        ['1', '3', 'A', 'B'],
        ['1', '4', 'A', 'C'],
        ['1', '5', 'A', 'B'],
        ['1', '6', 'B', 'C'],
        ['1', '7', 'A', 'A'],
        ['1', '8', 'C', 'C'],
        ['1', '9', 'C', 'C'],
        ['1', '10', 'A', 'A'],
        ['2', '1', 'AB', 'ACD'],
        ['2', '2', 'DF', 'BCD'],
        ['2', '3', 'AC', 'BCDE'],
        ['2', '4', 'ACD', 'BCDF'],
        ['2', '5', 'ACDE', 'ACD'],
      ];

      const worksheet = XLSX.utils.aoa_to_sheet(sampleXLSXData);
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, 'Sample Data');

      const blob = new Blob([XLSX.write(workbook, { bookType: 'xlsx', type: 'array' })], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });

      // Create download link to save file locally
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'answer_key.xlsx';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      console.log('Sample file downloaded. Please place it in the answer-keys folder.');
      setGradingMessage(
        'Sample file downloaded. Please place answer_key.xlsx in the answer-keys folder and try again.',
      );

      return true;
    } catch (error) {
      console.error('Error creating sample file:', error);
      setGradingMessage('Error creating sample file: ' + error.message);
      return false;
    }
  };

  // Handle file input changes
  const handlePdfFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setPdfFile(file);

      // Auto-load answer key when PDF is uploaded
      const answerKeyLoaded = await autoLoadAnswerKey();

      if (answerKeyLoaded) {
        set_valid_files(true);
      } else {
        set_valid_files(false);
      }
    } else {
      setPdfFile(null);
      set_valid_files(false);
      setGradingMessage('');
    }
  };

  const create_json_from_array = (xlsx_data) => {
    if (!xlsx_data || xlsx_data.length < 3) {
      throw new Error('File must have at least 3 rows');
    }

    // Validate header structure
    const firstRow = xlsx_data[0];
    const secondRow = xlsx_data[1];

    if (!firstRow || !secondRow) {
      throw new Error('Invalid file structure: Missing header rows');
    }

    // Check if first row contains ExamID or Assignment
    const firstCell = firstRow[0] ? firstRow[0].toString().toLowerCase() : '';
    if (!firstCell.includes('examid') && !firstCell.includes('assignment')) {
      throw new Error('Invalid file format: First row must start with "ExamID" or "Assignment"');
    }

    // Check if second row contains Part and Question
    if (
      !secondRow[0] ||
      !secondRow[1] ||
      !secondRow[0].toString().toLowerCase().includes('part') ||
      !secondRow[1].toString().toLowerCase().includes('question')
    ) {
      throw new Error('Invalid file format: Second row must contain "Part" and "Question"');
    }

    const results = [];
    for (let col = 2; col < xlsx_data[0].length; col++) {
      const raw_exam_id = xlsx_data[0][col];
      const exam_id = raw_exam_id !== undefined && raw_exam_id !== null ? String(raw_exam_id).trim() : '';
      if (!exam_id) break;

      const result = { exam_id: exam_id };

      let questionNumber = 1;
      for (let row = 2; row < xlsx_data.length && questionNumber <= TOTAL_NUM_QUESTIONS; row++) {
        const cell_raw = xlsx_data[row][col];
        result[String(questionNumber)] = cell_raw !== undefined && cell_raw !== null ? String(cell_raw).trim() : '';
        questionNumber++;
      }
      results.push(result);
    }

    if (results.length === 0) {
      throw new Error('No valid exam data found. Please check the file format.');
    }

    return results;
  };

  const handle_key_file_change = async (event) => {
    const file = event.target.files[0];
    if (file) {
      set_xlsx_file(file);
      setGradingMessage(pdfFile ? `Exam file: ${pdfFile.name}, Key file: ${file.name}` : `Key file: ${file.name}`);

      try {
        if (!file.name.toLowerCase().endsWith('.xlsx')) {
          throw new Error('Please upload an XLSX file.');
        }

        // Parse XLSX to 2D array
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const xlsxData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        // Convert to your JSON format
        const simpleJson = create_json_from_array(xlsxData);
        console.log('Parsed answer key JSON:', simpleJson);

        set_valid_files(true);
        localStorage.setItem('examarkAnswerKey', JSON.stringify(simpleJson));
        localStorage.setItem('examarkAnswerKeyFileName', file.name);
      } catch (error) {
        setGradingMessage(`Error parsing key file: ${error.message}`);
        console.error('Error parsing key file:', error);
      }
    } else {
      set_xlsx_file(null);
      set_valid_files(false);
      setGradingMessage(pdfFile ? `Exam file: ${pdfFile.name}` : '');
    }
  };

  // Start grading process
  const handle_grade_exam = async () => {
    if (!pdfFile) {
      setGradingMessage('Please upload the exam PDF file.');
      return;
    }

    if (!xlsx_file) {
      setGradingMessage('Please ensure at least one .xlsx file exists in the answer-keys folder.');
      return;
    }

    setGradingMessage(`Uploading and initiating grading for PDF: ${pdfFile.name} with auto-loaded answer key...`);
    setIsGrading(true);
    setIsGradingComplete(false);
    set_csv_data(null);
    setImages([]);
    setShowNavigationOptions(false);
    setProgress({
      stage: 'initializing',
      step: 'Starting upload...',
      currentPage: 0,
      totalPages: 0,
      progressPercent: 0.0,
    });

    try {
      // Get the parsed answer key JSON from localStorage
      const answerKeyJson = JSON.parse(localStorage.getItem('examarkAnswerKey') || '[]');
      if (answerKeyJson.length === 0) {
        throw new Error('Failed to parse answer key file');
      }

      const formData = new FormData();
      formData.append('pdfFile', pdfFile);
      const jsonBlob = new Blob([JSON.stringify(answerKeyJson)], {
        type: 'application/json',
      });
      formData.append('answerKey', jsonBlob, 'answer_key.json');

      const response = await fetch('http://localhost:8080/extract', {
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
      console.error('Error sending grading request:', error);
      setGradingMessage('An error occurred while communicating with the server. Please try again.');
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to start grading',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0,
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
        progressPercent: statusData.progress || 0.0,
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

      if (statusData.status === 'completed') {
        clearInterval(statusCheckInterval.current);
        setGradingMessage('Grading completed! Fetching results...');
        setIsGradingComplete(true);

        // Fetch results
        fetchResults(id);
      } else if (statusData.status === 'error') {
        clearInterval(statusCheckInterval.current);
        setGradingMessage(`Error: ${statusData.error || 'Grading failed'}`);
        setIsGrading(false);
        setProgress({
          stage: 'error',
          step: statusData.error || 'Grading failed',
          currentPage: 0,
          totalPages: 0,
          progressPercent: 0.0,
        });
      }
    } catch (error) {
      console.error('Error checking status:', error);
      setGradingMessage(`Error checking grading status: ${error.message}`);
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to check status',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0,
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
      set_csv_data(csvText);

      // Fetch image list from MinIO via backend - returns MinIO URLs
      const imagesResponse = await fetch(`http://localhost:8080/results/${id}/images`);
      if (!imagesResponse.ok) {
        throw new Error(`Failed to fetch image list: ${imagesResponse.status}`);
      }

      const imagesData = await imagesResponse.json();

      // Images now come with MinIO URLs directly
      const imageUrls = imagesData.images.map((img) => ({
        name: img.name,
        url: img.url, // Direct MinIO URL
      }));

      setImages(imageUrls);

      // Clear any previous edits
      localStorage.removeItem('examarkEdits');

      // Save results to localStorage for other pages
      localStorage.setItem('examarkJobId', id);
      localStorage.setItem('examarkCsvData', csvText);
      localStorage.setItem('examarkImages', JSON.stringify(imageUrls));

      // Save QR info if available
      if (imagesData.qrInfo) {
        localStorage.setItem('examarkQrInfo', imagesData.qrInfo);
      }

      setGradingMessage('Grading request completed successfully!');
      setIsGrading(false);
      setShowNavigationOptions(true);
      setProgress({
        stage: 'completed',
        step: 'All done!',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 100.0,
      });
    } catch (error) {
      console.error('Error fetching results:', error);
      setGradingMessage(`Error fetching results: ${error.message}`);
      setIsGrading(false);
      setProgress({
        stage: 'error',
        step: 'Failed to fetch results',
        currentPage: 0,
        totalPages: 0,
        progressPercent: 0.0,
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
          <div className="progress-step">{progress.step}</div>
          {progress.totalPages > 0 && (
            <div className="progress-pages">
              Page {progress.currentPage} of {progress.totalPages}
            </div>
          )}
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${Math.max(0, Math.min(100, progress.progressPercent))}%`,
            }}
          ></div>
        </div>
        <div className="progress-percentage">{Math.round(progress.progressPercent)}%</div>
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
              <p>Upload the exam papers (PDF) to begin grading.</p>

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

                {/* Answer key upload section is now hidden - auto-loaded from answer-keys folder */}
                <div className="file-upload-area" style={{ display: 'none' }}>
                  <button
                    type="button"
                    onClick={() => xlsx_input_ref.current && xlsx_input_ref.current.click()}
                    className="btn btn-info btn-large"
                    disabled={isGrading}
                  >
                    <i className="fas fa-file-xlsx"></i> Upload Answer XLSX
                  </button>
                  <input
                    type="file"
                    accept=".xlsx"
                    onChange={handle_key_file_change}
                    ref={xlsx_input_ref}
                    style={{ display: 'none' }}
                    id="xlsx-upload"
                    disabled={isGrading}
                  />
                  {xlsx_file && <span className="file-name">CSV: {xlsx_file.name}</span>}
                </div>
              </div>

              <button
                onClick={handle_grade_exam}
                className="btn btn-success btn-xl"
                disabled={!valid_files || isGrading}
              >
                {isGrading ? 'Grading in progress...' : 'Start Grading'}
              </button>

              <ProgressBar />

              {gradingMessage && <p className="grading-message">{gradingMessage}</p>}

              {/* CSV Helper Section - hidden when grading is in progress */}
              {!isGrading && !isGradingComplete && (
                <div className="csv-helper-section">
                  <div className="helper-info">
                    <h3>
                      <i className="fas fa-question-circle"></i> Answer Key Setup
                    </h3>
                    <p>Place any .xlsx answer key file in the answer-keys folder to enable automatic loading.</p>
                    <button type="button" onClick={download_sample_xlsx} className="btn btn-outline btn-medium">
                      <i className="fas fa-download"></i> Download Sample Key
                    </button>
                    <button
                      type="button"
                      onClick={create_sample_in_folder}
                      className="btn btn-outline btn-medium"
                      style={{ marginLeft: '10px' }}
                    >
                      <i className="fas fa-plus"></i> Create Sample in Folder
                    </button>
                  </div>
                  <div className="csv-format-info">
                    <h4>Key File Format Guidelines:</h4>
                    <ul>
                      <li>
                        <strong>Header Row:</strong> Contains ExamID and exam ids (101, 102, etc.)
                      </li>
                      <li>
                        <strong>Part Column:</strong> Indicates the section number (1, 2, etc.)
                      </li>
                      <li>
                        <strong>Question Column:</strong> Question number within each part
                      </li>
                      <li>
                        <strong>Key Columns:</strong> Correct answers for each exam version
                      </li>
                      <li>
                        <strong>Multiple Choice:</strong> Use single letters (A, B, C, D)
                      </li>
                      <li>
                        <strong>Multiple Selection:</strong> Use combinations (AB, ACD, BCDE)
                      </li>
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
                <button onClick={navigateToResults} className="nav-button results-button">
                  <i className="fas fa-list-alt"></i>
                  View Grading Results
                  <span className="description">Detailed view of each page with answers</span>
                </button>

                <button onClick={navigateToSheet} className="nav-button sheet-button">
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
                  set_csv_data(null);
                  setImages([]);
                  setPdfFile(null);
                  set_xlsx_file(null);
                  setShowNavigationOptions(false);
                  setGradingMessage('');
                  setProgress({
                    stage: '',
                    step: '',
                    currentPage: 0,
                    totalPages: 0,
                    progressPercent: 0.0,
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
