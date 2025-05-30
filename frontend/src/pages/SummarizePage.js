import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './SummarizePage.css';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

function SummarizePage() {
  const [csvData, setCsvData] = useState('');
  const [csvRows, setCsvRows] = useState([]);
  const [images, setImages] = useState([]);
  const [jobId, setJobId] = useState(null);

  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [hasResults, setHasResults] = useState(false);
  const [selectedCell, setSelectedCell] = useState({ row: 0, col: 0 });
  
  const [isEditing, setIsEditing] = useState(false);
  const [editedMetadata, setEditedMetadata] = useState({});
  const [editedAnswers, setEditedAnswers] = useState({});
  const [approved, setApproved] = useState(false);
  const tableRef = useRef(null);

  const [isGrading, setIsGrading] = useState(false);
  const [answerKey, setAnswerKey] = useState(Array(24).fill(''));
  const [pointValues, setPointValues] = useState(Array(24).fill(1));
  const [gradingResults, setGradingResults] = useState(null);

  // Load data on component mount
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
      
      // Parse CSV data
      let csvDataToUse = savedCsvData;
      let parsedRows = csvDataToUse.split('\n').map(line => 
        line.split(',').map(cell => cell.trim())
      );

      if (savedEdits) {
        const edits = JSON.parse(savedEdits);
        parsedRows = applyEditsToRows(parsedRows, edits);
        
        // Update CSV data with applied edits
        csvDataToUse = parsedRows.map(row => row.join(',')).join('\n');
        
        // Save the updated CSV back to localStorage
        localStorage.setItem('examarkCsvData', csvDataToUse);
        
        // Load edits into state for compatibility
        if (edits.metadata) setEditedMetadata(edits.metadata);
        if (edits.answers) setEditedAnswers(edits.answers);
      }
      
      setCsvData(csvDataToUse);
      setCsvRows(parsedRows);

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

// New helper function to apply edits to CSV rows
const applyEditsToRows = (rows, edits, images) => {
    if (!edits || !rows.length) return rows;
    
    const newRows = rows.map(row => [...row]); 
    
    // Apply metadata edits
    if (edits.metadata) {
      Object.entries(edits.metadata).forEach(([imageIndex, metadataObj]) => {
        // Find columns that correspond to this image
        const imgIdx = parseInt(imageIndex, 10);
        const startCol = 2 + imgIdx; // Assuming metadata starts at column 2
        
        // Apply each metadata edit
        Object.entries(metadataObj).forEach(([label, value]) => {
          // Find the row for this metadata item (typically rows 0-3)
          for (let i = 0; i < 4; i++) {
            if (newRows[i][0] === label || newRows[i][1] === label) {
              newRows[i][startCol] = value;
              break;
            }
          }
        });
      });
    }
    
    // Apply answer edits
    if (edits.answers) {
    Object.entries(edits.answers).forEach(([key, value]) => {
        // Parse key format: "[imageIndex]-[part]-[questionIdx]"
        const [imgIdx, part, questionIdx] = key.split('-');
        const imageIndex = parseInt(imgIdx, 10);
        
        // Map to the correct column in the CSV
        const colIndex = 2 + imageIndex; // Assuming data starts at column 2
        
        // Find the correct row based on part and question index
        let rowIndex = -1;
        
        if (part === '1') {
        // For part 1, find corresponding row (usually after row 3)
        for (let i = 4; i < newRows.length; i++) {
            if (newRows[i][0] === '1' && newRows[i][1] === questionIdx) {
            rowIndex = i;
            break;
            }
        }
        } else if (part === '2') {
        // For part 2, use a different approach based on your CSV structure
        const [qIdx, qRow] = questionIdx.split('-'); // if format is "qIdx-row"
        for (let i = 4; i < newRows.length; i++) {
            if (newRows[i][0] === '2' && newRows[i][1] === qIdx) {
            rowIndex = i + parseInt(qRow, 10); // Adjust if needed
            break;
            }
        }
        }
        
        // Apply the edit if we found the correct row
        if (rowIndex >= 0 && colIndex < newRows[rowIndex].length) {
        newRows[rowIndex][colIndex] = value;
        }
    });
    }
    
    // Update the state with the edited rows
    setCsvRows(newRows);
    return newRows;
};

// Updated saveChanges function
const saveChanges = () => {
    // First update the local state
    const updatedCsv = updateCsvFromRows();
    
    // Prepare edits in the format expected by ResultsPage
    const editsToSave = {
    metadata: {},
    answers: {}
    };
    
    // Extract edits by comparing with original CSV data
    const originalRows = csvData.split('\n')
    .map(line => line.split(',')
        .map(cell => cell.trim()));
    
    // Check each cell for changes
    csvRows.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
        // Skip first two columns (part and question)
        if (colIndex < 2) return;
        
        // Check if this is a header row (metadata)
        const isMetadata = rowIndex < 4;
        
        // Calculate which image this column belongs to
        const imageIndex = colIndex - 2; // Assuming data starts at column 2
        
        // Skip if out of range of original data
        if (originalRows.length <= rowIndex || originalRows[rowIndex].length <= colIndex) return;
        
        // Check if the value has changed
        if (cell !== originalRows[rowIndex][colIndex]) {
        if (isMetadata) {
            // It's metadata
            const label = row[0] || row[1]; // Use whichever column has the label
            if (!editsToSave.metadata[imageIndex]) {
            editsToSave.metadata[imageIndex] = {};
            }
            editsToSave.metadata[imageIndex][label] = cell;
        } else {
            // It's an answer
            const part = row[0];
            const questionIdx = row[1];
            const key = `${imageIndex}-${part}-${questionIdx}`;
            editsToSave.answers[key] = cell;
        }
        }
    });
    });
    
    // Save the edits to localStorage
    localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));
    
    // Update UI
    setIsEditing(false);
    alert('CSV data saved successfully!');
};

  // Parse CSV data into a rows array
  const parseCsvData = (csvString) => {
    if (!csvString) return;
    
    const rows = csvString.split('\n')
      .map(line => line.split(',')
        .map(cell => cell.trim()));
    
    setCsvRows(rows);
  };

  // Update CSV from edited rows
  const updateCsvFromRows = () => {
    const newCsvData = csvRows
      .map(row => row.join(','))
      .join('\n');
    
    setCsvData(newCsvData);
    localStorage.setItem('examarkCsvData', newCsvData);
    return newCsvData;
  };

  // Handle cell edit
  const handleCellEdit = (rowIndex, colIndex, newValue) => {
    if (!isEditing) return;
    
    const newRows = [...csvRows];
    newRows[rowIndex][colIndex] = newValue;
    setCsvRows(newRows);
  };

  // Handle input changes and auto uppercase
  const handleInput = (e, rowIndex, colIndex) => {
    if (!isEditing) return;
    
    let value = e.target.textContent;
    
    // Auto uppercase for answer cells (rows after headers, non-metadata columns)
    if (rowIndex > 3 && colIndex > 1) {
      if (value.length > 1) {
        value = value.charAt(0);
        e.target.textContent = value;
      }
      
      const uppercase = value.toUpperCase();
      if (uppercase !== value) {
        e.target.textContent = uppercase;
      }
    }
  };

  // Determine cell style based on content
  const getCellStyle = (value, rowIndex, colIndex) => {
    // Skip styling for header rows
    if (rowIndex < 4) return {};
    
    // Only apply to answer columns
    if (colIndex < 2) return {};

      let styles = {};

    // Check if this is one of the last 6 rows (grading results)
    const isGradingResultRow = csvRows.length > 0 && rowIndex >= csvRows.length - 7;
    
    if (isGradingResultRow) {
      if (value && /^\d+(\.\d+)?$/.test(value.toString())) {
        styles.fontWeight = 'bold';
        styles.fontSize = '18px';
        
        const rowLabel = csvRows[rowIndex]?.[1]?.toLowerCase() || '';
        
        if (rowLabel.includes('correct')) {
          // Blue background for correct answers
          styles.backgroundColor = '#e3f2fd';
          styles.color = '#1976d2';
          styles.border = '2px solid #2196f3';
        } else if (rowLabel.includes('points')) {
          // Green background for points
          styles.backgroundColor = '#e8f5e8';
          styles.color = '#2e7d32';
          styles.border = '2px solid #4caf50';
        }
      }
      
      return styles;
    }

    // Style lowercase letters 
    if (value && /[a-z]/.test(value)) {
      return { backgroundColor: 'cyan' };
    }
    
    // Style invalid answers
    // For Part 1, valid answers are A, B, C, D
    // For Part 2, valid answers are D, S
    const part = csvRows[rowIndex]?.[0] || '';
    if (value && value.trim() !== '') {
      if (part === '1') {
        // For Part 1, valid answers are A, B, C, D, _
        const validAnswers = ['A', 'B', 'C', 'D', '_'];
        if (!validAnswers.includes(value.toUpperCase())) {
          return { backgroundColor: 'yellow' };
        }
      } else if (part === '2') {
        // For Part 2, answers are strings containing only D, S, _
        const validPattern = /^[DS_]+$/i;
        if (!validPattern.test(value)) {
          return { backgroundColor: 'yellow' };
        }
      }
    }
    
    return {};
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

  // Handle image navigation
  const findImageForColumn = (colIndex) => {
    // Column 0, 1 might be metadata (e.g., Part, Question, etc.)
    // So column 2 corresponds to the first image (index 0)
    
    // Map column index to image index (adjusting for the metadata columns)
    const imageIndex = colIndex - 2;
    
    // Make sure we don't go beyond the available images
    if (imageIndex >= 0 && imageIndex < images.length) {
      return imageIndex;
    }
    
    // Return current index if we can't map properly
    return currentImageIndex;
  };
  
  // Update handleCellClick to use column mapping
  const handleCellClick = (rowIndex, colIndex) => {
    // Set the selected cell
    setSelectedCell({ row: rowIndex, col: colIndex });
    
    // Find and show the corresponding image based on column
    const imageIndex = findImageForColumn(colIndex);
    if (imageIndex !== currentImageIndex) {
      setCurrentImageIndex(imageIndex);
    }
  };

  // Render CSV table with proper frozen handling
  
  const renderCsvTable = () => {
    if (!csvRows.length) return <div>No CSV data available</div>;

    // Split rows: first 4 are frozen header rows, rest are body rows
    const headerRows = csvRows.slice(0, 4);
    const bodyRows = csvRows.slice(4);

    return (
      <table className="csv-table" ref={tableRef}>
        <thead className="csv-table-head">
          {headerRows.map((row, rowIndex) => (
            <tr key={`header-${rowIndex}`}>
              {row.map((cell, colIndex) => {
                const isSelected = selectedCell.row === rowIndex && selectedCell.col === colIndex;
                const isFrozenColumn = colIndex < 2;
                const isNonEditableRow = rowIndex === 0 || rowIndex === 3; // 1st and 4th rows
                const isEditable = isEditing && !isFrozenColumn && !isNonEditableRow;

                return (
                  <th key={colIndex}>
                    <div
                      id={`cell-${rowIndex}-${colIndex}`}
                      className={`csv-cell ${isEditable ? 'editable' : ''} ${isSelected ? 'current-cell' : ''} ${isFrozenColumn || isNonEditableRow ? 'column-frozen' : ''}`}
                      contentEditable={isEditable}
                      suppressContentEditableWarning={true}
                      onKeyDown={(e) => (isFrozenColumn || isNonEditableRow) ? null : handleKeyDown(e, rowIndex, colIndex)}
                      onInput={(e) => (isFrozenColumn || isNonEditableRow) ? null : handleInput(e, rowIndex, colIndex)}
                      onBlur={(e) => (isFrozenColumn || isNonEditableRow) ? null : handleCellEdit(rowIndex, colIndex, e.target.textContent)}
                      onClick={() => handleCellClick(rowIndex, colIndex)}
                      tabIndex={isEditable ? 0 : -1}
                      style={getCellStyle(cell, rowIndex, colIndex)}
                    >
                      {cell}
                    </div>
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {bodyRows.map((row, rowIndex) => {
            const actualRowIndex = rowIndex + 4; 
            return (
              <tr key={`body-${rowIndex}`}>
                {row.map((cell, colIndex) => {
                  const isSelected = selectedCell.row === actualRowIndex && selectedCell.col === colIndex;
                  const isFrozenColumn = colIndex < 2;
                  const isEditable = isEditing && !isFrozenColumn;

                  const hasLowercase = cell && /[a-z]/.test(cell);
                  const displayValue = hasLowercase ? cell.toUpperCase() : cell;

                  return (
                    <td key={colIndex}>
                      <div
                        id={`cell-${actualRowIndex}-${colIndex}`}
                        className={`csv-cell ${isEditable ? 'editable' : ''} ${isSelected ? 'current-cell' : ''} ${isFrozenColumn ? 'column-frozen' : ''}`}
                        contentEditable={isEditable}
                        suppressContentEditableWarning={true}
                        onKeyDown={(e) => !isFrozenColumn && handleKeyDown(e, actualRowIndex, colIndex)}
                        onInput={(e) => !isFrozenColumn && handleInput(e, actualRowIndex, colIndex)}
                        onBlur={(e) => !isFrozenColumn && handleCellEdit(actualRowIndex, colIndex, e.target.textContent)}
                        onClick={() => handleCellClick(actualRowIndex, colIndex)}
                        tabIndex={isEditable ? 0 : -1}
                        // In the body rows section of renderCsvTable
                      style={getCellStyle(cell, actualRowIndex, colIndex)}
                      >
                        {displayValue}
                      </div>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  };

  // Update handleKeyDown to skip non-editable cells
  const handleKeyDown = (e, rowIndex, colIndex) => {
    if (!isEditing) return;

    const rows = csvRows.length;
    const cols = csvRows[0]?.length || 0;
    let newRow = rowIndex;
    let newCol = colIndex;

    // Helper function to check if a cell is non-editable
    const isCellNonEditable = (row, col) => {
      return col < 2 || row === 0 || row === 3; // First 2 columns OR 1st row OR 4th row
    };

    switch (e.key) {
      case 'ArrowUp':
        newRow = Math.max(0, rowIndex - 1);
        // Skip non-editable rows
        while (newRow >= 0 && isCellNonEditable(newRow, colIndex)) {
          newRow--;
        }
        if (newRow < 0) newRow = rowIndex; // Stay in place if no valid cell above
        break;
        
      case 'ArrowDown':
      case 'Enter':
        newRow = Math.min(rows - 1, rowIndex + 1);
        // Skip non-editable rows
        while (newRow < rows && isCellNonEditable(newRow, colIndex)) {
          newRow++;
        }
        if (newRow >= rows) newRow = rowIndex; // Stay in place if no valid cell below
        break;
        
      case 'ArrowLeft':
        newCol = Math.max(0, colIndex - 1);
        // Skip frozen columns
        while (newCol >= 0 && isCellNonEditable(rowIndex, newCol)) {
          newCol--;
        }
        if (newCol < 0) newCol = colIndex; // Stay in place if no valid cell to left
        break;
        
      case 'ArrowRight':
        newCol = Math.min(cols - 1, colIndex + 1);
        break;
        
      case 'Tab':
        e.preventDefault();
        if (e.shiftKey) {
          // Move backwards
          newCol = colIndex - 1;
          newRow = rowIndex;
          
          // Find previous editable cell
          while ((newRow > 0 || newCol >= 0) && isCellNonEditable(newRow, newCol)) {
            newCol--;
            if (newCol < 0) {
              newRow--;
              newCol = cols - 1;
            }
          }
          
          // If we went too far, stay in current position
          if (newRow < 0) {
            newRow = rowIndex;
            newCol = colIndex;
          }
        } else {
          // Move forwards
          newCol = colIndex + 1;
          newRow = rowIndex;
          
          // Find next editable cell
          while ((newRow < rows - 1 || newCol < cols) && isCellNonEditable(newRow, newCol)) {
            newCol++;
            if (newCol >= cols) {
              newRow++;
              newCol = 0;
            }
          }
          
          // If we went too far, stay in current position
          if (newRow >= rows) {
            newRow = rowIndex;
            newCol = colIndex;
          }
        }
        break;
        
      default:
        return;
    }

    // Only move if we found a valid, editable cell
    if ((newRow !== rowIndex || newCol !== colIndex) && !isCellNonEditable(newRow, newCol)) {
      e.preventDefault();
      setSelectedCell({ row: newRow, col: newCol });
      const nextCell = document.getElementById(`cell-${newRow}-${newCol}`);
      if (nextCell) nextCell.focus();
    }
  };

  // Render the CSV table and image display
  const handleGradeExam = async () => {
    if (!jobId) {
      alert('No job ID available. Please re-upload the exam.');
      return;
    }
    
    setIsGrading(true);
    
    try {
      const response = await fetch('http://localhost:8080/grade-exam', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      // Get the updated CSV data
      const csvResponse = await fetch(`http://localhost:8080/results/${jobId}/csv`);
      if (!csvResponse.ok) {
        throw new Error(`Failed to fetch updated CSV: ${csvResponse.status}`);
      }
      
      const updatedCsvData = await csvResponse.text();
      setCsvData(updatedCsvData);
      parseCsvData(updatedCsvData);
      
      alert('Grading completed successfully!');
      
    } catch (error) {
      console.error('Error during grading:', error);
      alert(`Grading failed: ${error.message}`);
    } finally {
      setIsGrading(false);
    }
  };

  return (
    <div className="SummarizePage">
      {/* Header Section */}
      <header className="summarize-header">
        <div className="summarize-header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="summarize-header-logo" />
        </div>
        <div className="summarize-header-center">
          <h1>Exam Summary</h1>
          <p>Edit and review your exam results in spreadsheet format</p>
        </div>
        <div className="summarize-header-right">
          <img src={FamiLogo} alt="Fami Logo" className="summarize-header-fami-logo" />
        </div>
      </header>
      
      {hasResults ? (
        <>
          <div className="summarize-container">
            <div className="image-display">
              {images.length > 0 ? (
                <>
                  <img 
                    src={`http://127.0.0.1:8080/results/${jobId}/images/${images[currentImageIndex]}`} 
                    alt={`Exam page ${currentImageIndex + 1}`}
                    className="result-image"
                  />
                  <div className="summarize-image-navigation">
                    <button 
                      className="summarize-nav-button"
                      onClick={showPrevImage} 
                      disabled={currentImageIndex === 0}
                    >
                      Previous
                    </button>
                    <span className="summarize-nav-text">Page {currentImageIndex + 1} of {images.length}</span>
                    <button 
                      className="summarize-nav-button"
                      onClick={showNextImage} 
                      disabled={currentImageIndex === images.length - 1}
                    >
                      Next
                    </button>
                  </div>
                </>
              ) : (
                <div className="no-images">No images available</div>
              )}
            </div>
            
            <div className="csv-display">
              <div className="csv-controls">
                <h3>CSV Data Editor</h3>
                <div className="action-buttons">
                  {!isEditing ? (
                    <>  
                      <button className="btn btn-primary btn-small" onClick={() => setIsEditing(true)}>
                        EDIT CSV FILE
                      </button>
                      <button 
                        className="btn btn-success btn-small" 
                        onClick={handleGradeExam}
                        disabled={isGrading}
                      >
                        {isGrading ? 'GRADING...' : 'GRADE EXAM'}
                      </button>
                    </>
                  ) : (
                    <>
                      <button className="btn btn-success btn-small" onClick={saveChanges}>
                        Save Changes
                      </button>
                      <button className="btn btn-secondary btn-small" onClick={() => {
                        setIsEditing(false);
                        parseCsvData(csvData);
                      }}>
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </div>
              <div className="csv-table-container">
                {renderCsvTable()}
              </div>
              {isEditing && (
                <div className="editing-instructions">
                  <p>Navigation: Use arrow keys to move between cells. Press Enter to move down.</p>
                  <p>Editing: Answer cells are auto-converted to uppercase. Invalid answers are highlighted.</p>
                </div>
              )}

              <div className="action-footer">
                <button
                  className="btn btn-info btn-medium"
                  onClick={() => window.location.href = '/'}
                >
                  Back to Main
                </button>
                
                <Link to="/results">
                  <button className="btn btn-info btn-medium">View Results</button>
                </Link>
                
                <button 
                  className="btn btn-success btn-medium"
                  onClick={() => {
                    const csvContent = updateCsvFromRows();
                    const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
                    const link = document.createElement("a");
                    link.setAttribute("href", encodedUri);
                    link.setAttribute("download", "exam_results.csv");
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                  }}
                >
                  Download CSV
                </button>
                
                <button
                  className="btn btn-danger btn-medium"
                  onClick={() => {
                    if (window.confirm('Are you sure you want to clear all results data?')) {
                      localStorage.removeItem('examarkJobId');
                      localStorage.removeItem('examarkCsvData');
                      localStorage.removeItem('examarkImages');
                      setHasResults(false);
                      setCsvRows([]);
                      setImages([]);
                    }
                  }}
                >
                  Clear Results
                </button>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="no-results-message">
          <p>No data available. Please grade an exam first.</p>
          <Link to="/grade">
            <button className="btn btn-primary btn-large">Go to Grading Page</button>
          </Link>
        </div>
      )}
    </div>
  );
}
export default SummarizePage;