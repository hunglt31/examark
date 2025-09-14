import React, { useState, useRef, useEffect } from 'react';
import * as XLSX from 'xlsx';
import { Link } from 'react-router-dom';
import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';
import './GradingPage.css';

function GradeExamPage() {
  const [pdfFiles, setPdfFiles] = useState([]);
  const [xlsxData, setXlsxData] = useState([]);
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [gradingMessage, setGradingMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [jobInfos, setJobInfos] = useState([]);
  const [completedJobs, setCompletedJobs] = useState([]);
  const [isPolling, setIsPolling] = useState(false);
  const pdfInputRef = useRef(null);
  const xlsxInputRef = useRef(null);
  const pollingIntervalRef = useRef(null);
  const [validationError, setValidationError] = useState(null);

  // Handle XLSX file upload and extract user names and task numbers
  const handleXlsxUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const sheet = workbook.Sheets[sheetName];
      // Convert sheet to JSON array (each row is an array of values)
      const jsonData = XLSX.utils.sheet_to_json(sheet, { header: 1 });
      setXlsxData(jsonData);
      extractUsers(jsonData);
    };
    reader.readAsArrayBuffer(file);
  };

  // Simplified extractUsers function
  const extractUsers = (data) => {
    if (!data.length) return;

    const header = data[0];
    const nameIdx = header.findIndex((col) => col && col.toString().toLowerCase().includes('họ và tên'));
    const qrIdx = header.findIndex((col) => col && col.toString().toLowerCase().includes('mã qr'));

    if (nameIdx === -1) {
      setGradingMessage("Could not find 'Họ và tên' column in the XLSX file");
      return;
    }
    if (qrIdx === -1) {
      setGradingMessage("Warning: Could not find 'Mã QR' column in the XLSX file");
    }

    // Keep track of user counts and QR codes
    const userInfo = {};

    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      if (row && row[nameIdx]) {
        const userName = row[nameIdx].toString().trim();
        const qrCode = qrIdx !== -1 && row[qrIdx] ? row[qrIdx].toString().trim() : null;

        if (userInfo[userName]) {
          userInfo[userName].fileCount++;
          if (qrCode && !userInfo[userName].qrCodes.includes(qrCode)) {
            userInfo[userName].qrCodes.push(qrCode);
          }
        } else {
          userInfo[userName] = {
            name: userName,
            fileCount: 1,
            qrCodes: qrCode ? [qrCode] : [],
          };
        }
      }
    }

    const extractedUsers = Object.values(userInfo);
    setUsers(extractedUsers);
  };

  // Handle selection of user from the scroll
  const handleUserSelect = (event) => {
    setValidationError(null);
    const userName = event.target.value;
    setSelectedUser(users.find((u) => u.name === userName));
  };

  // Handle file input changes
  const handlePdfFileChange = async (event) => {
    setValidationError(null);
    const files = event.target.files;
    console.log('Selected files:', files);
    if (files && files.length > 0) {
      const newFilesArray = Array.from(files);
      console.log('New files array:', newFilesArray);

      setPdfFiles((prevFiles) => {
        const existingFileNames = prevFiles.map((f) => f.name);
        const uniqueNewFiles = newFilesArray.filter((file) => !existingFileNames.includes(file.name));
        const combinedFiles = [...prevFiles, ...uniqueNewFiles];
        console.log('Combined files:', combinedFiles);

        const fileNames = combinedFiles.map((f) => f.name);
        setGradingMessage(`Selected file(s): ${fileNames.join(', ')}`);

        return combinedFiles;
      });
    }

    event.target.value = '';
  };

  // Function to remove a specific file from the selection
  const removeFile = (indexToRemove) => {
    setPdfFiles((prevFiles) => {
      const updatedFiles = prevFiles.filter((_, index) => index !== indexToRemove);
      const fileNames = updatedFiles.map((f) => f.name);
      setGradingMessage(updatedFiles.length > 0 ? `Selected file(s): ${fileNames.join(', ')}` : '');
      return updatedFiles;
    });
  };

  // Function to clear all files
  const clearAllFiles = () => {
    setPdfFiles([]);
    setGradingMessage('');
  };

  // Open EventSource (SSE) for a given jobId to listen to progress updates
  const subscribeJobProgress = (jobId) => {
    const evtSource = new EventSource(`http://localhost:8080/events/${jobId}`);

    // Log when connection opens
    evtSource.onopen = () => {
      console.log(`EventSource connection opened for job ${jobId}`);
    };

    evtSource.onmessage = (e) => {
      console.log(`Job ${jobId} progress update:`, e.data);
      try {
        const data = JSON.parse(e.data);

        // Update job info in state
        setJobInfos((prevJobs) =>
          prevJobs.map((job) =>
            job.jobId === jobId
              ? {
                  ...job,
                  progress: data.progress || 0,
                  currentStage: data.currentStage || job.currentStage,
                  currentStep: data.currentStep || job.currentStep,
                  currentPage: data.currentPage || 0,
                  totalPages: data.totalPages || 0,
                  status: data.status || job.status,
                  errorMessage: data.error || job.errorMessage,
                }
              : job,
          ),
        );

        // Close connection if job is complete or has error
        if (data.status === 'completed' || data.status === 'error') {
          console.log(`Job ${jobId} finished with status: ${data.status}`);
          evtSource.close();
        }
      } catch (err) {
        console.error(`Error parsing SSE data for job ${jobId}:`, err, 'Raw data:', e.data);
      }
    };

    evtSource.onerror = (e) => {
      console.error(`EventSource error for job ${jobId}:`, e);
      evtSource.close();
    };

    // Return the evtSource to allow closing it if needed
    return evtSource;
  };

  // Function to fetch CSV data from URL
  const fetchCSVData = async (csvUrl) => {
    try {
      const response = await fetch(csvUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch CSV: ${response.status}`);
      }
      const csvText = await response.text();
      return csvText;
    } catch (error) {
      console.error('Error fetching CSV data:', error);
      return null;
    }
  };

  // Poll for completed results from the session endpoint
  const pollSessionResults = async (sessionId) => {
    try {
      const response = await fetch(`http://localhost:8080/results/session/${sessionId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session results: ${response.status}`);
      }

      const sessionData = await response.json();
      console.log('Session results:', sessionData);

      if (sessionData.data && sessionData.data.length > 0) {
        const jobsWithCSVData = await Promise.all(
          sessionData.data.map(async (result) => {
            if (result.csv && result.status === 'completed') {
              const csvData = await fetchCSVData(result.csv);
              return {
                ...result,
                csvData: csvData,
              };
            }
            return result;
          }),
        );

        setCompletedJobs(jobsWithCSVData);

        // Check if all jobs are completed
        const allJobsCompleted = jobInfos.every((job) =>
          jobsWithCSVData.some((result) => result.jobId === job.jobId && result.status === 'completed'),
        );

        if (allJobsCompleted && jobsWithCSVData.length === jobInfos.length) {
          setIsPolling(false);
          setGradingMessage('All grading tasks completed successfully!');
        }
      }
    } catch (error) {
      console.error('Error polling session results:', error);
      setGradingMessage(`Error fetching results: ${error.message}`);
    }
  };

  // Separate useEffect for polling
  useEffect(() => {
    if (sessionId && isPolling) {
      pollingIntervalRef.current = setInterval(() => {
        pollSessionResults(sessionId);
      }, 10000);

      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
        }
      };
    }
  }, [sessionId, isPolling]);

  // Separate useEffect to check if all jobs are completed
  useEffect(() => {
    if (jobInfos.length > 0 && completedJobs.length > 0) {
      const allJobsCompleted = jobInfos.every((job) =>
        completedJobs.some((result) => result.jobId === job.jobId && result.status === 'completed'),
      );

      if (allJobsCompleted && completedJobs.length === jobInfos.length) {
        setIsPolling(false);
        setGradingMessage('All grading tasks completed successfully!');
      }
    }
  }, [jobInfos, completedJobs]);

  // Start grading process
  const handleGradeExam = async () => {
    if (!selectedUser) {
      setGradingMessage('Please select a user.');
      return;
    }
    const requiredTasks = selectedUser.taskCount;
    if (pdfFiles.length < requiredTasks) {
      setGradingMessage(`Not enough files uploaded. Required: ${requiredTasks}, uploaded: ${pdfFiles.length}`);
      return;
    }

    setGradingMessage(`Starting grading for ${pdfFiles.length} file(s)...`);

    const formData = new FormData();
    pdfFiles.forEach((file) => {
      formData.append('pdfFiles', file);
    });

    const qrInfo = {};
    if (selectedUser.qrCodes && selectedUser.qrCodes.length > 0) {
      selectedUser.qrCodes.forEach((qr, idx) => {
        qrInfo[`qr${idx + 1}`] = qr;
      });
    }
    formData.append('qr-info', JSON.stringify(qrInfo));

    try {
      const response = await fetch('http://localhost:8080/extract', {
        method: 'POST',
        body: formData,
      });

      const resultJson = await response.json();
      console.log('Extract API response:', resultJson);

      if (!response.ok) {
        // Handle error response
        if (resultJson.invalid_files) {
          setValidationError({
            message: resultJson.message || 'QR code validation failed',
            invalidFiles: resultJson.invalid_files,
          });
        } else {
          setGradingMessage(`Error: ${resultJson.error || 'Unknown error occurred'}`);
        }
        return;
      }

      const serverSessionId = resultJson.metadata?.sessionId;
      if (!serverSessionId) {
        setGradingMessage('Error: No session ID returned from server');
        return;
      }

      const initialJobs = resultJson.data.map((job) => ({
        ...job,
        jobId: job.jobId,
        progress: 0,
        status: 'processing',
        currentStep: 'Initializing...',
        currentPage: 0,
        totalPages: 0,
      }));

      console.log('Setting up jobs:', initialJobs);
      setJobInfos(initialJobs);
      setSessionId(serverSessionId);
      setIsPolling(true);
      setGradingMessage('Grading tasks launched. Listening for progress updates...');

      // Subscribe to each job's progress updates with a slight delay between each
      initialJobs.forEach((job, index) => {
        // Add a small delay between each subscription to avoid overwhelming the server
        setTimeout(() => {
          console.log(`Subscribing to job ${job.jobId} progress updates`);
          subscribeJobProgress(job.jobId);
        }, index * 200);
      });
    } catch (error) {
      console.error('Error during grading process:', error);
      setGradingMessage(`Error grading exams: ${error.message}`);
    }
  };

  // Save completed jobs to localStorage when ready
  React.useEffect(() => {
    if (completedJobs.length > 0) {
      const examData = {
        metadata: {
          totalPDFs: completedJobs.length,
          timestamp: new Date().toISOString(),
        },
        data: completedJobs,
      };
      localStorage.setItem('examData', JSON.stringify(examData));
    }
  }, [completedJobs]);

  // Cleanup polling on component unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Progress bar component for individual jobs
  const JobProgressBar = ({ job }) => {
    const progress = typeof job.progress === 'number' ? job.progress : 0;
    const status = job.status || 'processing';
    const currentStep = job.currentStep || 'Initializing...';

    console.log('JobProgressBar props:', { progress, status, currentStep, job });

    return (
      <div className="job-progress-container">
        <div className="job-progress-header">
          <strong>{job.pdf}</strong>
          <span className={`status-badge ${status}`}>{status}</span>
        </div>

        <div className="job-progress-info">
          <div className="progress-step">{currentStep}</div>
          {job.currentStage && (
            <div className="progress-stage">Stage: {job.currentStage.replace('_', ' ').toUpperCase()}</div>
          )}
          {job.totalPages > 0 && (
            <div className="progress-pages">
              Page {job.currentPage || 0} of {job.totalPages}
            </div>
          )}
        </div>

        <div className="progress-bar-container">
          <div className="progress-bar">
            <div
              className={`progress-fill ${status}`}
              style={{
                width: `${Math.max(0, Math.min(100, progress))}%`,
                transition: 'width 0.3s ease-in-out',
              }}
            />
          </div>
          <div className="progress-percentage">{Math.round(progress)}%</div>
        </div>

        {job.errorMessage && (
          <div className="error-message">
            <i className="fas fa-exclamation-triangle"></i>
            {job.errorMessage}
          </div>
        )}
      </div>
    );
  };

  // Component to display QR validation errors
  const QRValidationError = ({ message, invalidFiles }) => {
    return (
      <div className="qr-validation-error">
        <div className="error-header">
          <i className="fas fa-exclamation-triangle"></i>
          <h4>{message}</h4>
        </div>
        <div className="invalid-files-list">
          <h5>Invalid Files:</h5>
          <ul>
            {invalidFiles.map((file, idx) => (
              <li key={idx} className="invalid-file-item">
                <strong>{file.filename}</strong>
                <span className="error-reason">{file.error}</span>
                {file.qr_code && <span className="qr-code">QR Code: {file.qr_code}</span>}
              </li>
            ))}
          </ul>
          <div className="error-help">
            <p>Please make sure to upload files with QR codes that match the selected user's assigned classes.</p>
            <ul>
              {selectedUser && selectedUser.qrCodes && selectedUser.qrCodes.length > 0 && (
                <li>
                  <strong>Allowed QR codes for {selectedUser.name}:</strong>
                  <ul>
                    {selectedUser.qrCodes.map((qr, idx) => (
                      <li key={idx}>{qr}</li>
                    ))}
                  </ul>
                </li>
              )}
            </ul>
          </div>
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
          <Link to="/" className="back-link-header">
            <button className="btn btn-secondary btn-medium">Back to Main Page</button>
          </Link>
          <img src={FamiLogo} alt="Fami Logo" className="grade-header-fami-logo" />
        </div>
      </header>

      <div className="exam-container">
        <div className="exam-content">
          {jobInfos.length === 0 ? (
            <div className="two-column-layout">
              {/* Left Column - Task Assignment */}
              <div className="left-column">
                <div className="task-assignment-section">
                  <h3>Task Assignment</h3>
                  <div className="task-upload">
                    <button
                      type="button"
                      onClick={() => xlsxInputRef.current && xlsxInputRef.current.click()}
                      className="btn btn-primary btn-medium"
                    >
                      <i className="fas fa-file-excel"></i> Upload Task XLSX
                    </button>
                    <input
                      type="file"
                      accept=".xlsx,.xls"
                      onChange={handleXlsxUpload}
                      ref={xlsxInputRef}
                      style={{ display: 'none' }}
                    />
                  </div>

                  {users.length > 0 && (
                    <div className="user-selection">
                      <label htmlFor="user-select">Select Grader:</label>
                      <select
                        id="user-select"
                        onChange={handleUserSelect}
                        value={selectedUser ? selectedUser.name : ''}
                        className="user-select-dropdown"
                      >
                        <option value="" disabled>
                          -- Select User --
                        </option>
                        {users.map((user) => (
                          <option key={user.name} value={user.name}>
                            {user.name} ({user.fileCount} files required)
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  {selectedUser && (
                    <div className="user-tasks-info">
                      <h4>{selectedUser.name}</h4>
                      <p>
                        <strong>Required PDF files:</strong> {selectedUser.fileCount}
                      </p>
                      <p className="upload-status">
                        <strong>Current status:</strong> {pdfFiles.length} of {selectedUser.fileCount} files uploaded
                        {pdfFiles.length >= selectedUser.fileCount ? (
                          <span className="status-complete"> ✓ Complete</span>
                        ) : (
                          <span className="status-incomplete"> ⚠️ Incomplete</span>
                        )}
                      </p>
                      {selectedUser.qrCodes && selectedUser.qrCodes.length > 0 && (
                        <div className="user-qr-codes">
                          <p>
                            <strong>Classes to grade:</strong>
                          </p>
                          <ul className="qr-list">
                            {selectedUser.qrCodes.map((qr, idx) => (
                              <li key={idx}>{qr}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column - PDF Upload */}
              <div className="right-column">
                <div className="file-upload-section">
                  <h3>Upload Exam PDFs</h3>
                  <p>Upload the exam papers (PDF) to begin grading.</p>
                  <div className="file-upload-area">
                    <button
                      type="button"
                      onClick={() => pdfInputRef.current && pdfInputRef.current.click()}
                      className="btn btn-primary btn-large"
                    >
                      <i className="fas fa-file-pdf"></i> Upload Exam PDF(s)
                    </button>
                    <input
                      type="file"
                      accept=".pdf"
                      multiple
                      onChange={handlePdfFileChange}
                      ref={pdfInputRef}
                      style={{ display: 'none' }}
                    />
                    {/* Display selected files with remove option */}
                    {pdfFiles.length > 0 && (
                      <div className="selected-files">
                        <div className="files-header">
                          <span className="file-count">Selected {pdfFiles.length} PDF(s):</span>
                          <button type="button" onClick={clearAllFiles} className="btn btn-danger btn-small">
                            Clear All
                          </button>
                        </div>
                        <ul className="file-list">
                          {pdfFiles.map((file, index) => (
                            <li key={index} className="file-item">
                              <span className="file-name">{file.name}</span>
                              <button
                                type="button"
                                onClick={() => removeFile(index)}
                                className="btn btn-danger btn-tiny"
                              >
                                ×
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="results-section">
              <h3>Grading Progress</h3>
              {sessionId && (
                <p>
                  <strong>Session ID:</strong> {sessionId}
                </p>
              )}

              {jobInfos.length > 0 && (
                <div className="jobs-section">
                  <h3>Job Progress</h3>
                  {jobInfos.map((job) => (
                    <JobProgressBar key={job.jobId} job={job} />
                  ))}
                </div>
              )}

              {/* Overall progress summary */}
              <div className="overall-progress">
                <p>
                  <strong>Overall Progress:</strong> {completedJobs.length} of {jobInfos.length} jobs completed
                </p>
                {isPolling && (
                  <p>
                    <em>Polling for results every 10 seconds...</em>
                  </p>
                )}
              </div>

              {completedJobs.length > 0 && (
                <div className="completed-results">
                  <h3>Completed Grading - View Results</h3>
                  <ul>
                    {completedJobs.map((result, idx) => (
                      <li key={idx} className="completed-job">
                        <div className="job-result-info">
                          <Link to={`/sheet?jobId=${encodeURIComponent(result.pdf)}`}>
                            {result.pdf} - View Grading Sheet
                          </Link>
                          {result.csvData ? (
                            <span className="csv-status success">✓ CSV Data Loaded</span>
                          ) : result.csv ? (
                            <span className="csv-status loading">⏳ Loading CSV...</span>
                          ) : (
                            <span className="csv-status error">⚠ No CSV Available</span>
                          )}
                        </div>
                        {result.csvData && (
                          <details className="csv-preview">
                            <summary>Preview CSV Data</summary>
                            <pre className="csv-content">
                              {result.csvData.split('\n').slice(0, 10).join('\n')}
                              {result.csvData.split('\n').length > 10 && '\n... (showing first 10 lines)'}
                            </pre>
                          </details>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {/* Common elements that should appear below both columns */}
          <div className="bottom-section">
            {/* Add validation error display here */}
            {validationError && (
              <QRValidationError message={validationError.message} invalidFiles={validationError.invalidFiles} />
            )}

            {/* Display regular status messages */}
            {gradingMessage && <p className="grading-message">{gradingMessage}</p>}

            <button onClick={handleGradeExam} className="btn btn-success btn-xl" disabled={pdfFiles.length === 0}>
              Start Grading
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GradeExamPage;
