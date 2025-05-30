import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './ResultsPage.css';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

function ResultsPage() {
  const [csvData, setCsvData] = useState(null);
  const [images, setImages] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [hasResults, setHasResults] = useState(false);
  // Store edits per image index
  const [editedMetadata, setEditedMetadata] = useState({});
  const [editedAnswers, setEditedAnswers] = useState({});
  const [approved, setApproved] = useState(false);

  useEffect(() => {
    // Get URL parameters to check if we should force refresh
    const queryParams = new URLSearchParams(window.location.search);
    const shouldRefresh = queryParams.get('refresh');
    
    // Load data from localStorage
    const savedJobId = localStorage.getItem('examarkJobId');
    const savedCsvData = localStorage.getItem('examarkCsvData');
    const savedImages = localStorage.getItem('examarkImages');
    const savedEdits = localStorage.getItem('examarkEdits');
    
    if (savedJobId && savedCsvData && savedImages) {
      setJobId(savedJobId);
      setCsvData(savedCsvData);

      const parsedImages = JSON.parse(savedImages);
      const sortedImages = parsedImages.sort((a, b) => {
        // Extract page numbers (e.g., from "page_1.jpg" extract 1)
        const pageNumA = parseInt(a.match(/page_(\d+)/)?.[1] || '0', 10);
        const pageNumB = parseInt(b.match(/page_(\d+)/)?.[1] || '0', 10);
        return pageNumA - pageNumB;
      });

      setImages(sortedImages);
      setHasResults(true);

      // Load any saved edits from localStorage
      if (savedEdits) {
        const edits = JSON.parse(savedEdits);
        if (edits.metadata) setEditedMetadata(edits.metadata);
        if (edits.answers) setEditedAnswers(edits.answers);
      }
      
      // Reset editing state when data is refreshed
      if (shouldRefresh) {
        setApproved(false);
        
        // Remove the refresh parameter from URL without page reload
        const newUrl = window.location.pathname;
        window.history.replaceState({}, document.title, newUrl);
      }
    }
  }, [window.location.search]);

  // Handle metadata edit using the current image index 
  const handleMetadataEdit = (label, newValue) => {
    const updatedMetadata = {
      ...editedMetadata,
      [currentImageIndex]: {
        ...editedMetadata[currentImageIndex],
        [label]: newValue
      }
    };
    
    setEditedMetadata(updatedMetadata);
    
    // Save to localStorage
    const editsToSave = {
      metadata: updatedMetadata,
      answers: editedAnswers
    };
    localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));
  };

  // Modify handleAnswerEdit to save to localStorage
  const handleAnswerEdit = (part, questionIdx, newValue) => {
    if (!approved) {
      const key = `${currentImageIndex}-${part}-${questionIdx}`;
      const updatedAnswers = {
        ...editedAnswers,
        [key]: newValue.toUpperCase()
      };
      
      setEditedAnswers(updatedAnswers);
      
      // Save to localStorage
      const editsToSave = {
        metadata: editedMetadata,
        answers: updatedAnswers
      };
      localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));
    }
  };

  // Get the current answer value (either edited or original)
  const getAnswerValue = (part, questionIdx, originalAnswer) => {
    const key = `${currentImageIndex}-${part}-${questionIdx}`;
    return editedAnswers[key] !== undefined ? editedAnswers[key] : originalAnswer;
  };

  // Navigate through images
  const showNextImage = () => {
    if (currentImageIndex < images.length - 1) {
      setCurrentImageIndex(currentImageIndex + 1);
    }
  };
  
  const showPrevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1);
    }
  };
  
  // Parse CSV to get results for current image
  const getCurrentImageResults = () => {
    if (!csvData || images.length === 0) return { metadata: [], part1: [], part2: [] };
    
    // Parse CSV
    const rows = csvData.split('\n');
    if (rows.length < 4) return { metadata: [], part1: [], part2: [] };
    
    // Get the current image filename (e.g., "page_0.jpg")
    const currentImageName = images[currentImageIndex]; 
    const baseImageName = currentImageName.split('.')[0]; // Remove extension (page_0)
    
    // Find which column corresponds to our current image
    const headerRow = rows[0].split(',');
    let imageColumnIndex = -1;
    
    // Try to find exact match for page_X in headers
    for (let i = 0; i < headerRow.length; i++) {
      if (headerRow[i].trim() === baseImageName) {
        imageColumnIndex = i;
        break;
      }
    }
    
    // If not found, try to determine based on the page number
    if (imageColumnIndex === -1 && baseImageName.startsWith('page_')) {
      const pageNum = parseInt(baseImageName.split('_')[1], 10);
      imageColumnIndex = pageNum + 2; // Assuming page_0 is at column index 2
    }
    
    // If still not found, return empty
    if (imageColumnIndex === -1 || imageColumnIndex >= headerRow.length) {
      return { metadata: [], part1: [], part2: [] };
    }
    
    // Prepare result containers
    const metadata = [];
    const part1 = [];
    const part2 = [];
    
    // Add metadata (Student ID, Exam ID)
    if (rows.length >= 3) {
      // Student ID row
      const studentIdRow = rows[1].split(',');
      if (studentIdRow.length > imageColumnIndex) {
        metadata.push({
          label: "Student ID",
          value: studentIdRow[imageColumnIndex].trim()
        });
      }
      
      // Exam ID row
      const examIdRow = rows[2].split(',');
      if (examIdRow.length > imageColumnIndex) {
        metadata.push({
          label: "Exam ID",
          value: examIdRow[imageColumnIndex].trim()
        });
      }
    }
    
    // Find the Part/Question header row
    let questionHeaderRow = 3;
    for (let i = 3; i < rows.length; i++) {
      const cells = rows[i].split(',');
      if (cells.length > 1 && cells[0].trim() === 'Part' && cells[1].trim() === 'Question') {
        questionHeaderRow = i;
        break;
      }
    }
    
    // Process answer rows after the header, sorting by part
    for (let i = questionHeaderRow + 1; i < rows.length; i++) {
      const row = rows[i].split(',');
      if (row.length > imageColumnIndex) {
        const part = row[0].trim();
        const question = row[1].trim();
        const answer = row[imageColumnIndex].trim();
        
        if (part && question) {
          const item = {
            question: question,
            answer: answer
          };
          
          if (part === '1') {
            part1.push(item);
          } else if (part === '2') {
            part2.push(item);
          }
        }
      }
    }
    
    return { metadata, part1, part2 };
  };

  // Render the answer with editable field
  const renderAnswer = (part, questionIdx, answer, cellId) => {
    const currentAnswer = getAnswerValue(part, questionIdx, answer);
    let style = {};

    const allowedAnswers = part === '1' ? ['A', 'B', 'C', 'D', '_'] : ['D', 'S', '_'];

    if (answer && answer.match(/[a-z]/)) {
      style = { backgroundColor: 'cyan' }; 
    } else if (!allowedAnswers.includes(currentAnswer)) {
      style = { backgroundColor: 'yellow' };
    }
    
    const onInputHandler = (e) => {
      let text = e.target.textContent;
      if (text.length > 1) {
        text = text.slice(0, 1);
      }
      // convert to uppercase automatically
      const upperText = text.toUpperCase();
      if (upperText !== e.target.textContent) {
        e.target.textContent = upperText;
      }
    };

    const onKeyDownHandler = (e) => {
      if (cellId) {
        // Expect cellId format: "cell-<part>-<col>-<row>"
        const idParts = cellId.split('-');
        const col = parseInt(idParts[2], 10);
        const row = parseInt(idParts[3], 10);
        let numCols, numRows;
        if (part === '1') {
          numCols = 4; 
          numRows = 4; 
        } else {
          numCols = 8; 
          numRows = 6; 
        }
        let nextCol = col;
        let nextRow = row;

        switch (e.key) {
          case 'Enter':
          case 'ArrowDown':
            if (row < numRows - 1) {
              nextRow = row + 1;
            } else if (col < numCols - 1) {
              nextRow = 0;
              nextCol = col + 1;
            }
            break;
          case 'ArrowUp':
            if (row > 0) {
              nextRow = row - 1;
            } else if (col > 0) {
              nextCol = col - 1;
              nextRow = numRows - 1;
            }
            break;
          case 'ArrowRight':
            if (col < numCols - 1) {
              nextCol = col + 1;
            }
            break;
          case 'ArrowLeft':
            if (col > 0) {
              nextCol = col - 1;
            }
            break;
          default:
            return;
        }
        const nextCell = document.getElementById(`cell-${part}-${nextCol}-${nextRow}`);
        if (nextCell) {
          e.preventDefault();
          nextCell.focus();
        }
      }
    };

    const cellProps = {};
    if (!approved && cellId) {
      cellProps.id = cellId;
      cellProps.onKeyDown = onKeyDownHandler;
      cellProps.onInput = onInputHandler;
    }

    return (
      <span
        contentEditable={!approved}
        data-placeholder="X"
        style={style}
        onBlur={(e) => handleAnswerEdit(part, questionIdx, e.target.textContent)}
        suppressContentEditableWarning={true}
        {...cellProps}
      >
        {currentAnswer ? currentAnswer.toUpperCase() : ''}
      </span>
    );
  };

  return (
    <div className="ResultsPage">
      {hasResults ? (
        <>
          {/* Header Section */}
          <header className="results-header">
            <div className="results-header-left">
              <img src={UniversityLogo} alt="HUST Logo" className="results-header-logo" />
            </div>
            <div className="results-header-center">
              <h1>Exam Results</h1>
              <p>Review and edit your graded exam results</p>
            </div>
            <div className="results-header-right">
              <img src={FamiLogo} alt="Fami Logo" className="results-header-fami-logo" />
            </div>
          </header>

          <div className="bottom-right-buttons">
            <Link to="/">
              <button className="results-btn results-btn-primary">Back to Main</button>
            </Link>
            <Link to="/summarize">
             <button className="results-btn results-btn-primary">View Summary</button>
            </Link>
            <button 
              className="results-btn results-btn-secondary"
              onClick={() => window.open(`http://127.0.0.1:8080/results/${jobId}/csv`, '_blank')}
            >
              Download CSV
            </button>
            <button
              className="results-btn results-btn-danger"
              onClick={() => {
                localStorage.removeItem('examarkJobId');
                localStorage.removeItem('examarkCsvData');
                localStorage.removeItem('examarkImages');
                setHasResults(false);
              }}
            >
              Clear Results
            </button>
          </div>    

          <div className="results-container">
            <div className="results-display">
              <div className="result-image-display">
                {images.length > 0 ? (
                  <>
                    <img 
                      src={`http://127.0.0.1:8080/results/${jobId}/images/${images[currentImageIndex]}`} 
                      alt={`Graded exam page ${currentImageIndex + 1}`}
                      className="result-image"
                    />
                    <div className="results-image-navigation">
                      <div className="results-image-navigation-buttons">
                        <button 
                          className="results-nav-button"
                          onClick={showPrevImage} 
                          disabled={currentImageIndex === 0}
                        >
                          Previous
                        </button>
                        <span className="results-nav-text">Page {currentImageIndex + 1} of {images.length}</span>
                        <button 
                          className="results-nav-button"
                          onClick={showNextImage} 
                          disabled={currentImageIndex === images.length - 1}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="no-images">No images available</div>
                )}
              </div>
                <div className="results-text">
                {(() => {
                  const results = getCurrentImageResults();
                  return (
                    <>
                      {/* Metadata and Action buttons in one row */}
                      <div className="metadata-action-row">
                        <div className="metadata-inline">
                          {results.metadata.map((item, index) => (
                            <div className="metadata-item-inline" key={`meta-${index}`}>
                              <strong>{item.label}:</strong>{" "}
                              {(item.label === "Student ID" || item.label === "Exam ID") ? (
                                <span
                                  contentEditable
                                  suppressContentEditableWarning
                                  onBlur={(e) => handleMetadataEdit(item.label, e.target.textContent.trim())}
                                >
                                  {(editedMetadata[currentImageIndex] && editedMetadata[currentImageIndex][item.label]) || item.value}
                                </span>
                              ) : (
                                item.value
                              )}
                            </div>
                          ))}
                        </div>
                        <div className="action-buttons-inline">
                          {!approved ? (
                            <button className="approve-button" onClick={() => setApproved(true)}>
                              Approve Exam
                            </button>
                          ) : (
                            <button className="edit-button" onClick={() => setApproved(false)}>
                              Edit Exam
                            </button>
                          )}
                        </div>
                      </div>
                      {/* Part 1 */}
                      <div className="part-label">
                        <strong>Part 1</strong>
                      </div>
                      <div className="part1-grid">
                        {[0, 1, 2, 3].map((colIndex) => (
                          <div className="part1-column" key={`col-${colIndex}`}>
                            {results.part1.slice(colIndex * 4, colIndex * 4 + 4).map((item, rowIndex) => {
                              const flatIndex = colIndex * 4 + rowIndex;
                              const cellId = `cell-1-${colIndex}-${rowIndex}`;
                              return (
                                <div className="question-item" key={`p1-${cellId}`}>
                                  <strong>Q{item.question}:</strong> {renderAnswer('1', flatIndex, item.answer, cellId)}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                      {/* Part 2 */}
                      <div className="part-label">
                        <strong>Part 2</strong>
                      </div>
                      <div className="part2-table">
                        <table>
                          <thead>
                            <tr>
                              {results.part2.map((item, qIndex) => (
                                <th key={`p2-h-${qIndex}`}>{`Q${item.question}`}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {Array.from({ length: 6 }).map((_, rowIndex) => (
                              <tr key={`p2-row-${rowIndex}`}>
                                {results.part2.map((item, qIndex) => {
                                  const answerLetter =
                                    item.answer && item.answer.length > rowIndex
                                      ? item.answer[rowIndex]
                                      : 'X';
                                  const cellId = `cell-2-${qIndex}-${rowIndex}`;
                                  return (
                                    <td key={`p2-${qIndex}-${rowIndex}`}>
                                      {renderAnswer('2', `${qIndex}-${rowIndex}`, answerLetter, cellId)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {approved && (
                        <div className="approval-status">
                          <strong>Approved</strong>
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="no-results-message">
          <p>No exam results available. Please grade an exam first.</p>
          <Link to="/grade">
            <button className="start-grading-button">Go to Grading Page</button>
          </Link>
        </div>
      )}
    </div>
  );
}

export default ResultsPage;