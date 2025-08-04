import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './SheetPage.css';
import CustomAlert from '../components/CustomAlert';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

import BackArrowIcon from '../assets/icons/back-arrow.png';
import ResultIcon from '../assets/icons/result.png';
import DownloadIcon from '../assets/icons/download.png';
import DeleteIcon from '../assets/icons/delete.png';

import NextIcon from '../assets/icons/next.png';
import PreviousIcon from '../assets/icons/previous.png';

function SheetPage() {
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

  const [isRegrade, setIsRegrade] = useState(false);
  const [regradeJobId, setRegradeJobId] = useState(null);
  const [answerKeyFile, setAnswerKeyFile] = useState(null);
  const [showRegradeModal, setShowRegradeModal] = useState(false);

  const answerKeyInputRef = useRef(null);

  const [hasPreviousAnswerKey, setHasPreviousAnswerKey] = useState(false);
  const [previousAnswerKeyFileName, setPreviousAnswerKeyFileName] = useState('');
  const [showRegradeOptions, setShowRegradeOptions] = useState(false);

  // Add alert state
  const [alert, setAlert] = useState({
    isOpen: false,
    message: '',
    type: 'info',
    showConfirm: false,
    onConfirm: null,
  });

  const showAlert = (message, type = 'info', showConfirm = false, onConfirm = null) => {
    setAlert({
      isOpen: true,
      message,
      type,
      showConfirm,
      onConfirm,
    });
  };

  const closeAlert = () => {
    setAlert({
      isOpen: false,
      message: '',
      type: 'info',
      showConfirm: false,
      onConfirm: null,
    });
  };

  const handleClearSheet = () => {
    showAlert('Do you want to clear all sheet data? This action cannot be undone.', 'warning', true, () => {
      localStorage.removeItem('examarkJobId');
      localStorage.removeItem('examarkCsvData');
      localStorage.removeItem('examarkImages');
      localStorage.removeItem('examarkEdits');
      setHasResults(false);
      setCsvRows([]);
      setImages([]);
      closeAlert();
    });
  };

  // [MinIO] Load data on component mount
  useEffect(() => {
    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const urlJobId = urlParams.get('jobId');

    // Get data from localStorage
    const savedJobId = localStorage.getItem('examarkJobId') || urlJobId;
    const savedCsvData = localStorage.getItem('examarkCsvData');
    const savedImages = localStorage.getItem('examarkImages');

    const savedAnswerKey = localStorage.getItem('examarkAnswerKey');
    const savedFileName = localStorage.getItem('examarkAnswerKeyFileName');

    if (savedAnswerKey) {
      setHasPreviousAnswerKey(true);
      setPreviousAnswerKeyFileName(savedFileName || 'Previous answer key');
    }

    if (savedJobId && savedCsvData && savedImages) {
      setJobId(savedJobId);

      // Parse CSV data
      const rows = savedCsvData.split('\n').map((row) => {
        const cells = [];
        let current = '';
        let inQuotes = false;

        for (let i = 0; i < row.length; i++) {
          const char = row[i];
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            cells.push(current.trim());
            current = '';
          } else {
            current += char;
          }
        }
        cells.push(current.trim());
        return cells;
      });

      setCsvData(savedCsvData);
      setCsvRows(rows);

      const parsedImages = JSON.parse(savedImages);

      // Images should already have MinIO URLs from the backend
      const processedImages = parsedImages.map((img) => {
        if (typeof img === 'string') {
          // Old format - shouldn't happen with MinIO but keep for safety
          console.warn("Old image format detected, this shouldn't happen with MinIO");
          return {
            name: img,
            url: `http://localhost:8080/results/${savedJobId}/images/${img}`,
          };
        } else {
          // New format from MinIO - use direct URL
          return {
            name: img.name,
            url: img.url, // Direct MinIO URL
          };
        }
      });

      // Sort images by page number
      const sortedImages = processedImages.sort((a, b) => {
        const pageNumA = parseInt(a.name.match(/page_(\d+)/)?.[1] || '0', 10);
        const pageNumB = parseInt(b.name.match(/page_(\d+)/)?.[1] || '0', 10);
        return pageNumA - pageNumB;
      });

      setImages(sortedImages);
      setHasResults(true);
    }

    // *** THÊM: Reload data khi page được focus lại ***
    const handlePageFocus = () => {
      console.log('Page focused, reloading data...');
      const savedCsvData = localStorage.getItem('examarkCsvData');
      if (savedCsvData && savedCsvData !== csvData) {
        setCsvData(savedCsvData);
        // Parse lại CSV rows
        const newRows = savedCsvData.split('\n').map((row) => {
          const cells = [];
          let current = '';
          let inQuotes = false;

          for (let i = 0; i < row.length; i++) {
            const char = row[i];
            if (char === '"') {
              inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
              cells.push(current.trim());
              current = '';
            } else {
              current += char;
            }
          }
          cells.push(current.trim());
          return cells;
        });
        setCsvRows(newRows);
        console.log('CSV data updated from localStorage');
      }
    };

    // *** THÊM: Listen for storage changes from other tabs/pages ***
    const handleStorageChange = (e) => {
      if (e.key === 'examarkCsvData' && e.newValue) {
        setCsvData(e.newValue);
        console.log('CSV data updated from another page');
      } else if (e.key === 'examarkEdits' && e.newValue) {
        const edits = JSON.parse(e.newValue);
        if (edits.metadata) setEditedMetadata(edits.metadata);
        if (edits.answers) setEditedAnswers(edits.answers);
        console.log('Edits updated from another page');
      }
    };

    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [csvData]);

  // New helper function to apply edits to CSV rows
  const applyEditsToRows = (rows, edits) => {
    if (!edits || !rows.length) return rows;

    const newRows = rows.map((row) => [...row]);

    // Apply metadata edits
    if (edits.metadata) {
      Object.entries(edits.metadata).forEach(([imageIndex, metadataObj]) => {
        const imgIdx = parseInt(imageIndex, 10);

        // Map image index to CSV column - this should match ResultsPage logic
        const csvColumnIndex = imgIdx + 2; // Assuming metadata starts at column 2

        // Apply each metadata edit
        Object.entries(metadataObj).forEach(([label, value]) => {
          // Find the row for this metadata item
          for (let i = 0; i < Math.min(4, newRows.length); i++) {
            // Check both first and second columns for the label
            if (newRows[i][0] === label || newRows[i][1] === label) {
              if (csvColumnIndex < newRows[i].length) {
                newRows[i][csvColumnIndex] = value;
              }
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

        // Map image index to CSV column
        const csvColumnIndex = imageIndex + 2; // Same mapping as metadata

        // Find the correct row based on part and question index
        let rowIndex = -1;

        if (part === '1') {
          // For part 1, find the row with matching part and question
          const flatQuestionIndex = parseInt(questionIdx, 10);
          const actualQuestionNumber = flatQuestionIndex + 1; // Convert 0-based to 1-based

          for (let i = 4; i < newRows.length; i++) {
            if (newRows[i][0] === '1' && parseInt(newRows[i][1]) === actualQuestionNumber) {
              rowIndex = i;
              break;
            }
          }
        } else if (part === '2') {
          // For part 2, handle the format "qIdx-row"
          const [qIdx, qRow] = questionIdx.split('-');
          const questionNumber = parseInt(qIdx, 10) + 1; // Convert 0-based to 1-based
          const rowOffset = parseInt(qRow, 10);

          // Find the base row for this question in part 2
          for (let i = 4; i < newRows.length; i++) {
            if (newRows[i][0] === '2' && parseInt(newRows[i][1]) === questionNumber) {
              rowIndex = i;
              break;
            }
          }

          // For part 2, we need to handle multi-character answers
          // When editing in SheetPage, replace the entire answer string
          if (rowIndex >= 0 && csvColumnIndex < newRows[rowIndex].length) {
            // In SheetPage, we want to replace the entire answer string
            // So we'll use the value directly as the new answer
            newRows[rowIndex][csvColumnIndex] = value;
          }

          // Skip the normal assignment below
          rowIndex = -1;
        }

        // Apply the edit for part 1 or if we didn't handle it above
        if (rowIndex >= 0 && csvColumnIndex < newRows[rowIndex].length) {
          newRows[rowIndex][csvColumnIndex] = value;
        }
      });
    }

    return newRows;
  };

  const saveChanges = async () => {
    // First update the local state and get the updated CSV
    const updatedCsv = updateCsvFromRows();

    // Prepare edits in the format expected by ResultsPage
    const editsToSave = {
      metadata: {},
      answers: {},
    };

    // Extract edits by comparing with original CSV data
    const originalRows = csvData.split('\n').map((line) => line.split(',').map((cell) => cell.trim()));

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
            // It's an answer - need to convert CSV format to ResultsPage format
            const part = row[0];
            const questionNumber = parseInt(row[1], 10);

            if (part === '1') {
              // For Part 1: convert 1-based question number to 0-based index
              const questionIdx = questionNumber - 1;
              const key = `${imageIndex}-${part}-${questionIdx}`;
              editsToSave.answers[key] = cell;
            } else if (part === '2') {
              // For Part 2: need to handle multi-character answers
              // Each character in the answer string corresponds to a different row in ResultsPage
              const questionIdx = questionNumber - 1; // Convert to 0-based

              // Split the cell value into individual characters
              const answerString = cell || '';
              for (let charIndex = 0; charIndex < answerString.length; charIndex++) {
                const char = answerString[charIndex];
                if (char && char.trim() !== '') {
                  const key = `${imageIndex}-${part}-${questionIdx}-${charIndex}`;
                  editsToSave.answers[key] = char;
                }
              }
            }
          }
        }
      });
    });

    localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));

    window.dispatchEvent(
      new StorageEvent('storage', {
        key: 'examarkCsvData',
        newValue: updatedCsv,
        oldValue: csvData,
      }),
    );

    window.dispatchEvent(
      new StorageEvent('storage', {
        key: 'examarkEdits',
        newValue: JSON.stringify(editsToSave),
        oldValue: localStorage.getItem('examarkEdits'),
      }),
    );

    setCsvData(updatedCsv);
    setIsEditing(false);

    // Upload updated CSV to MinIO via backend
    try {
      const formData = new FormData();
      const csvBlob = new Blob([updatedCsv], { type: 'text/csv' });
      formData.append('csvFile', csvBlob, 'updated_results.csv');

      const response = await fetch(`http://127.0.0.1:8080/upload-csv/${jobId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to upload CSV: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      showAlert('CSV data saved successfully and uploaded to MinIO!', 'success');
    } catch (error) {
      console.error('Error uploading CSV to MinIO:', error);
      showAlert(`CSV saved locally but failed to upload to MinIO: ${error.message}`, 'warning');
    }
  };

  // Parse CSV data into a rows array
  const parseCsvData = (csvString) => {
    if (!csvString) return;

    const rows = csvString.split('\n').map((line) => line.split(',').map((cell) => cell.trim()));

    setCsvRows(rows);
  };

  // Update CSV from edited rows
  const updateCsvFromRows = () => {
    const newCsvData = csvRows.map((row) => row.join(',')).join('\n');

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

    // For Student ID (row 1) and Exam ID (row 2), only allow numbers
    if ((rowIndex === 1 || rowIndex === 2) && colIndex > 1) {
      const numbersOnly = value.replace(/\D/g, '');
      if (numbersOnly !== value) {
        e.target.textContent = numbersOnly;
        // Move cursor to end
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(e.target);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      return;
    }

    // Auto uppercase for answer cells (rows after headers, non-metadata columns)
    if (rowIndex > 3 && colIndex > 1) {
      const part = csvRows[rowIndex]?.[0] || '';

      if (part === '2') {
        // For Part 2: limit to 6 characters, allow any letters (like Part 1)
        const validChars = value.replace(/[^A-Za-z]/g, '');
        const limitedValue = validChars.slice(0, 6).toUpperCase();

        if (limitedValue !== value) {
          e.target.textContent = limitedValue;
          // Move cursor to end
          const range = document.createRange();
          const sel = window.getSelection();
          range.selectNodeContents(e.target);
          range.collapse(false);
          sel.removeAllRanges();
          sel.addRange(range);
        }
      } else {
        // For Part 1: single character only
        if (value.length > 1) {
          value = value.charAt(0);
          e.target.textContent = value;
        }

        const uppercase = value.toUpperCase();
        if (uppercase !== value) {
          e.target.textContent = uppercase;
        }
      }
    }
  };

  const shouldHighlightColumn = (colIndex) => {
    if (colIndex < 2 || !csvRows.length) return false;

    let part1UnderscoreCount = 0;
    let part2EmptyCount = 0;
    let part1Total = 0;
    let part2Total = 0;

    csvRows.forEach((row, rowIndex) => {
      if (rowIndex < 4) return;

      const part = row[0];
      const cellValue = row[colIndex] || '';

      if (part === '1') {
        part1Total++;
        if (cellValue.trim() === '_') {
          part1UnderscoreCount++;
        }
      } else if (part === '2') {
        part2Total++;
        if (cellValue.trim() === '______' || cellValue.trim() === '') {
          part2EmptyCount++;
        }
      }
    });

    // Highlight if more than 50% of answers are missing/empty
    const part1Threshold = part1Total > 0 ? part1UnderscoreCount / part1Total > 0.5 : false;
    const part2Threshold = part2Total > 0 ? part2EmptyCount / part2Total > 0.5 : false;

    return part1Threshold && part2Threshold;
  };

  // Determine cell style based on content
  const getCellStyle = (value, rowIndex, colIndex) => {
    const shouldHighlight = shouldHighlightColumn(colIndex);

    if (rowIndex < 4) {
      if ((rowIndex === 1 || rowIndex === 2) && colIndex > 1) {
        if (value && !isValidId(value)) {
          return { backgroundColor: 'yellow' };
        }
      }

      if (shouldHighlight && colIndex > 1) {
        return { backgroundColor: '#ffeb3b', border: '2px solid #ff9800' };
      }

      return {};
    }

    // Only apply to answer columns
    if (colIndex < 2) return {};

    let styles = {};

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
    let bodyRows = csvRows.slice(4);

    // Filter out the scoring rows (last 3 rows: PART 1 CORRECT, PART 2 CORRECT, TOTAL POINTS)
    bodyRows = bodyRows.filter((row, index) => {
      const rowLabel = row[1]?.toLowerCase() || '';
      return !rowLabel.includes('correct') && !rowLabel.includes('points');
    });

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
                      onKeyDown={(e) =>
                        isFrozenColumn || isNonEditableRow ? null : handleKeyDown(e, rowIndex, colIndex)
                      }
                      onInput={(e) => (isFrozenColumn || isNonEditableRow ? null : handleInput(e, rowIndex, colIndex))}
                      onBlur={(e) =>
                        isFrozenColumn || isNonEditableRow
                          ? null
                          : handleCellEdit(rowIndex, colIndex, e.target.textContent)
                      }
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
                        onBlur={(e) =>
                          !isFrozenColumn && handleCellEdit(actualRowIndex, colIndex, e.target.textContent)
                        }
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

    const selection = window.getSelection();
    const isEditingContent =
      selection.rangeCount > 0 && selection.getRangeAt(0).startContainer.nodeType === Node.TEXT_NODE;

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
        // Allow left arrow to move within cell content if actively editing
        if (isEditingContent) {
          const range = selection.getRangeAt(0);
          if (range.startOffset > 0) {
            return; // Let browser handle cursor movement within cell
          }
        }

        newCol = Math.max(0, colIndex - 1);
        // Skip frozen columns
        while (newCol >= 0 && isCellNonEditable(rowIndex, newCol)) {
          newCol--;
        }
        if (newCol < 0) newCol = colIndex; // Stay in place if no valid cell to left
        break;

      case 'ArrowRight':
        // Allow right arrow to move within cell content if actively editing
        if (isEditingContent) {
          const range = selection.getRangeAt(0);
          const textContent = range.startContainer.textContent || '';
          if (range.startOffset < textContent.length) {
            return; // Let browser handle cursor movement within cell
          }
        }

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

  const handleSaveExcel = () => {
    try {
      const currentCsvData = updateCsvFromRows();

      if (!currentCsvData) {
        showAlert('No data to save', 'error');
        return;
      }

      // Parse CSV data into rows
      const csvRows = currentCsvData.split('\n').filter((row) => row.trim());
      const data = csvRows.map((row) => row.split(',').map((cell) => cell.trim()));

      // Create simple Excel XML format
      let excelXML = `<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
 <Worksheet ss:Name="Exam Results">
  <Table>`;

      // Add data rows
      data.forEach((row) => {
        excelXML += '<Row>';
        row.forEach((cell) => {
          const cellValue = cell
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
          excelXML += `<Cell><Data ss:Type="String">${cellValue}</Data></Cell>`;
        });
        excelXML += '</Row>';
      });

      excelXML += `  </Table>
 </Worksheet>
</Workbook>`;

      // Create blob with Excel MIME type for XLSX
      const blob = new Blob([excelXML], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;

      // Use QR info for filename if available, otherwise use jobId
      const qrInfo = localStorage.getItem('examarkQrInfo');
      let filename;
      if (qrInfo) {
        // Clean QR info for filename (replace spaces with underscores)
        const cleanQrInfo = qrInfo.replace(/\s+/g, '_');
        filename = `${cleanQrInfo}.xlsx`;
      } else {
        filename = `exam_results_${jobId || 'sheet'}.xlsx`;
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error saving Excel:', error);
      showAlert('Error saving Excel file', 'error');
    }
  };

  // Helper function to parse CSV content to JSON
  const parseCsvToJson = (csvContent) => {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',').map((cell) => cell.trim());

    const result = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map((cell) => cell.trim());
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });
      result.push(row);
    }

    return result;
  };

  // Helper function to parse XLS content to JSON
  const parseXlsToJson = (xlsContent) => {
    // Simple XLS XML parser for our specific format
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xlsContent, 'text/xml');

    const rows = xmlDoc.querySelectorAll('Row');
    if (rows.length === 0) {
      throw new Error('Invalid XLS format');
    }

    // Extract headers from first row
    const headerCells = rows[0].querySelectorAll('Cell Data');
    const headers = Array.from(headerCells).map((cell) => cell.textContent.trim());

    const result = [];
    for (let i = 1; i < rows.length; i++) {
      const cells = rows[i].querySelectorAll('Cell Data');
      const row = {};
      headers.forEach((header, index) => {
        const cellValue = cells[index] ? cells[index].textContent.trim() : '';
        row[header] = cellValue;
      });
      result.push(row);
    }

    return result;
  };

  // Handle answer key file selection
  const handleAnswerKeyFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Update to accept XLS and CSV files
    if (
      file.type !== 'application/vnd.ms-excel' &&
      file.type !== 'text/csv' &&
      !file.name.toLowerCase().endsWith('.xls') &&
      !file.name.toLowerCase().endsWith('.csv')
    ) {
      showAlert('Please select an XLS or CSV file for the answer key.', 'error');
      return;
    }

    try {
      setIsRegrade(true);

      // Get current CSV data with any edits applied
      const currentCsvData = updateCsvFromRows();

      if (!currentCsvData) {
        showAlert('No CSV data available for re-grading.', 'error');
        setIsRegrade(false);
        return;
      }

      // Read and parse the answer key file to JSON
      let answerKeyJson;

      if (file.name.toLowerCase().endsWith('.xls')) {
        // For XLS files, read as text and parse XML
        const xlsContent = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = (e) => reject(e);
          reader.readAsText(file);
        });
        answerKeyJson = parseXlsToJson(xlsContent);
      } else {
        // For CSV files, read as text and parse CSV
        const csvContent = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = (e) => reject(e);
          reader.readAsText(file);
        });
        answerKeyJson = parseCsvToJson(csvContent);
      }

      // Save the parsed JSON to localStorage for future use
      localStorage.setItem('examarkAnswerKey', JSON.stringify(answerKeyJson));
      localStorage.setItem('examarkAnswerKeyFileName', file.name);
      setHasPreviousAnswerKey(true);
      setPreviousAnswerKeyFileName(file.name);

      // Prepare JSON payload
      const regradePayload = {
        jobId: jobId,
        csvData: currentCsvData,
        answerKey: answerKeyJson,
      };

      console.log('Sending regrade request with answer key JSON:', regradePayload);

      // Send re-grade request with JSON headers
      const response = await fetch('http://127.0.0.1:8080/regrade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(regradePayload),
      });

      if (!response.ok) {
        throw new Error(`Re-grading failed: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      const newRegradeJobId = result.jobId;
      setRegradeJobId(newRegradeJobId);

      // Poll for re-grade completion
      await pollRegradeStatus(newRegradeJobId);
    } catch (error) {
      console.error('Regrade error:', error);
      setIsRegrade(false);
      showAlert(`Re-grading failed: ${error.message}`, 'error');
    } finally {
      // Clear the file input for next use
      if (answerKeyInputRef.current) {
        answerKeyInputRef.current.value = '';
      }
    }
  };

  const handleRegradeWithExistingKey = async () => {
    setShowRegradeOptions(false);

    const savedAnswerKey = localStorage.getItem('examarkAnswerKey');

    if (!savedAnswerKey) {
      showAlert('No previous answer key found. Please upload a new one.', 'error');
      setShowRegradeModal(true);
      return;
    }

    try {
      setIsRegrade(true);

      // Get current CSV data with any edits applied
      const currentCsvData = updateCsvFromRows();

      if (!currentCsvData) {
        showAlert('No CSV data available for re-grading.', 'error');
        setIsRegrade(false);
        return;
      }

      // Parse the saved answer key JSON
      const answerKeyJson = JSON.parse(savedAnswerKey);

      // Prepare JSON payload with existing answer key
      const regradePayload = {
        jobId: jobId,
        csvData: currentCsvData,
        answerKey: answerKeyJson,
      };

      console.log('Sending regrade request with existing answer key JSON:', regradePayload);

      // Send re-grade request
      const response = await fetch('http://127.0.0.1:8080/regrade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(regradePayload),
      });

      if (!response.ok) {
        throw new Error(`Re-grading failed: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      const newRegradeJobId = result.jobId;
      setRegradeJobId(newRegradeJobId);

      // Poll for re-grade completion
      await pollRegradeStatus(newRegradeJobId);
    } catch (error) {
      console.error('Regrade error:', error);
      setIsRegrade(false);
      showAlert(`Re-grading failed: ${error.message}`, 'error');
    }
  };

  // // Handle answer key file selection
  // const handleAnswerKeyFileChange = async (event) => {
  //   const file = event.target.files[0];
  //   if (!file) return;

  //   if (file.type !== 'text/csv' && !file.name.toLowerCase().endsWith('.csv')) {
  //     showAlert('Please select a CSV file for the answer key.', 'error');
  //     return;
  //   }

  //   try {
  //     setIsRegrade(true);

  //     // Get current CSV data with any edits applied
  //     const currentCsvData = updateCsvFromRows();

  //     if (!currentCsvData) {
  //       showAlert('No CSV data available for re-grading.', 'error');
  //       setIsRegrade(false);
  //       return;
  //     }

  //     // Read the answer key file content as text
  //     const answerKeyContent = await new Promise((resolve, reject) => {
  //       const reader = new FileReader();
  //       reader.onload = (e) => resolve(e.target.result);
  //       reader.onerror = (e) => reject(e);
  //       reader.readAsText(file);
  //     });

  //     // Save the new answer key to localStorage for future use
  //     localStorage.setItem('examarkAnswerKey', answerKeyContent);
  //     localStorage.setItem('examarkAnswerKeyFileName', file.name);
  //     setHasPreviousAnswerKey(true);
  //     setPreviousAnswerKeyFileName(file.name);

  //     // Prepare JSON payload
  //     const regradePayload = {
  //       jobId: jobId,
  //       csvData: currentCsvData,
  //       answerKey: answerKeyContent
  //     };

  //     console.log('Sending regrade request with new key:', regradePayload);

  //     // Send re-grade request with JSON headers
  //     const response = await fetch('http://127.0.0.1:8080/regrade', {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //         'Accept': 'application/json'
  //       },
  //       body: JSON.stringify(regradePayload)
  //     });

  //     if (!response.ok) {
  //       const errorText = await response.text();
  //       console.error('Regrade failed:', errorText);
  //       throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
  //     }

  //     const result = await response.json();
  //     console.log('Regrade response:', result);

  //     if (result.regrade_job_id) {
  //       await pollRegradeStatus(result.regrade_job_id);
  //     } else {
  //       throw new Error('No regrade job ID returned');
  //     }

  //   } catch (error) {
  //     console.error('Regrade error:', error);
  //     setIsRegrade(false);
  //     showAlert(`Re-grading failed: ${error.message}`, 'error');
  //   } finally {
  //     // Clear the file input for next use
  //     if (answerKeyInputRef.current) {
  //       answerKeyInputRef.current.value = '';
  //     }
  //   }
  // };

  // Handle re-grade button click
  const handleRegradeClick = () => {
    if (!jobId) {
      showAlert('No job ID found. Cannot perform re-grading.', 'error');
      return;
    }

    if (hasPreviousAnswerKey) {
      setShowRegradeOptions(true);
    } else {
      setShowRegradeModal(true);
    }
  };

  // Handle regrade with new answer key
  const handleRegradeWithNewKey = () => {
    setShowRegradeOptions(false);
    if (answerKeyInputRef.current) {
      answerKeyInputRef.current.click();
    }
  };

  // const handleRegradeWithExistingKey = async () => {
  //   setShowRegradeOptions(false);

  //   const savedAnswerKey = localStorage.getItem('examarkAnswerKey');
  //   if (!savedAnswerKey) {
  //     showAlert('No previous answer key found. Please upload a new one.', 'error');
  //     setShowRegradeModal(true);
  //     return;
  //   }

  //   try {
  //     setIsRegrade(true);

  //     // Get current CSV data with any edits applied
  //     const currentCsvData = updateCsvFromRows();

  //     if (!currentCsvData) {
  //       showAlert('No CSV data available for re-grading.', 'error');
  //       setIsRegrade(false);
  //       return;
  //     }

  //     // Prepare JSON payload with existing answer key
  //     const regradePayload = {
  //       jobId: jobId,
  //       csvData: currentCsvData,
  //       answerKey: savedAnswerKey
  //     };

  //     console.log('Sending regrade request with existing key:', regradePayload);

  //     // Send re-grade request
  //     const response = await fetch('http://127.0.0.1:8080/regrade', {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //         'Accept': 'application/json'
  //       },
  //       body: JSON.stringify(regradePayload)
  //     });

  //     if (!response.ok) {
  //       const errorText = await response.text();
  //       console.error('Regrade failed:', errorText);
  //       throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
  //     }

  //     const result = await response.json();
  //     console.log('Regrade response:', result);

  //     if (result.regrade_job_id) {
  //       await pollRegradeStatus(result.regrade_job_id);
  //     } else {
  //       throw new Error('No regrade job ID returned');
  //     }

  //   } catch (error) {
  //     console.error('Regrade error:', error);
  //     setIsRegrade(false);
  //     showAlert(`Re-grading failed: ${error.message}`, 'error');
  //   }
  // };

  const handleRegradeOptionsCancel = () => {
    setShowRegradeOptions(false);
  };

  // Add the polling function if not already present:
  const pollRegradeStatus = async (regradeJobId) => {
    const maxAttempts = 30; // 30 seconds timeout
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const statusResponse = await fetch(`http://127.0.0.1:8080/status/${regradeJobId}`);
        if (!statusResponse.ok) {
          throw new Error('Failed to check regrade status');
        }

        const statusData = await statusResponse.json();
        console.log('Regrade status:', statusData);

        if (statusData.status === 'completed') {
          setIsRegrade(false);
          showAlert('Re-grading completed successfully! Fetching updated results...', 'success');

          // Fetch updated results
          try {
            const updatedCsvResponse = await fetch(`http://127.0.0.1:8080/results/${jobId}/csv`);
            if (updatedCsvResponse.ok) {
              const updatedCsvData = await updatedCsvResponse.text();

              // Update the state with new data
              setCsvData(updatedCsvData);
              const newRows = updatedCsvData.split('\n').map((line) => line.split(',').map((cell) => cell.trim()));
              setCsvRows(newRows);

              // Update localStorage with new data
              localStorage.setItem('examarkCsvData', updatedCsvData);

              showAlert('Results updated successfully!', 'success');
            } else {
              throw new Error('Failed to fetch updated results');
            }
          } catch (error) {
            console.error('Error fetching updated results:', error);
            showAlert('Re-grading completed but failed to fetch updated results. Please refresh the page.', 'warning');
          }
          return;
        } else if (statusData.status === 'error') {
          throw new Error(statusData.error || 'Regrade failed');
        }

        // Wait 1 second before next check
        await new Promise((resolve) => setTimeout(resolve, 1000));
        attempts++;
      } catch (error) {
        console.error('Status check error:', error);
        setIsRegrade(false);
        throw error;
      }
    }

    setIsRegrade(false);
    throw new Error('Regrade timeout - process took too long');
  };

  // Cancel re-grade modal
  const handleRegradeCancel = () => {
    setShowRegradeModal(false);
    setAnswerKeyFile(null);
    if (answerKeyInputRef.current) {
      answerKeyInputRef.current.value = '';
    }
  };

  const isValidId = (value) => {
    return /^\d*$/.test(value.toString().trim());
  };

  return (
    <div className="SheetPage">
      <CustomAlert
        isOpen={alert.isOpen}
        message={alert.message}
        type={alert.type}
        onClose={closeAlert}
        showConfirm={alert.showConfirm}
        onConfirm={alert.onConfirm}
        confirmText="Clear Data"
        cancelText="Cancel"
      />

      {/* Header Section */}
      <header className="page-header">
        <div className="page-header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="page-header-logo" draggable="false" />
        </div>
        <div className="page-header-center">
          <h1>Exam Results Sheet</h1>
          <p>Review and edit results of all exams in spreadsheet format</p>
        </div>
        <div className="page-header-right">
          <div className="header-buttons">
            <Link to="/" draggable="false">
              <button className="header-btn header-btn-primary">
                Dashboard
                <img src={BackArrowIcon} alt="Back" className="header-btn-icon" draggable="false" />
              </button>
            </Link>
            <Link to="/results" draggable="false">
              <button className="header-btn header-btn-primary">
                Review Exam
                <img src={ResultIcon} alt="Result" className="header-btn-icon" draggable="false" />
              </button>
            </Link>
            <button className="header-btn header-btn-secondary" onClick={handleSaveExcel}>
              Save Excel
              <img src={DownloadIcon} alt="Download" className="header-btn-icon" draggable="false" />
            </button>
            <button className="header-btn header-btn-danger" onClick={handleClearSheet}>
              Clear Sheet
              <img src={DeleteIcon} alt="Delete" className="header-btn-icon" draggable="false" />
            </button>
          </div>
          <img src={FamiLogo} alt="Fami Logo" className="page-header-fami-logo" draggable="false" />
        </div>
      </header>

      {hasResults ? (
        <>
          {/* Image Container */}
          <div className="image-container">
            {images.length > 0 ? (
              <>
                <img
                  //src={`http://127.0.0.1:8080/results/${jobId}/images/${images[currentImageIndex]}`}
                  src={images[currentImageIndex].url}
                  alt={`Exam page ${currentImageIndex + 1}`}
                  className="result-image"
                  onError={(e) => {
                    console.error('Failed to load image from MinIO:', e.target.src);
                    e.target.src =
                      'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y4ZjlmYSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2Yzc1N2QiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pjwvc3ZnPg==';
                  }}
                />
                <div className="sheet-image-navigation">
                  <div className="sheet-image-navigation-buttons">
                    <button className="sheet-nav-button" onClick={showPrevImage} disabled={currentImageIndex === 0}>
                      <img src={PreviousIcon} alt="Previous" className="nav-icon" draggable="false" />
                    </button>
                    <span className="sheet-nav-text">
                      Exam {} {currentImageIndex + 1} of {images.length} - Page{' '}
                      {images[currentImageIndex]?.name.match(/page_(\d+)/)?.[1] || currentImageIndex + 1}
                    </span>
                    <button
                      className="sheet-nav-button"
                      onClick={showNextImage}
                      disabled={currentImageIndex === images.length - 1}
                    >
                      <img src={NextIcon} alt="Next" className="nav-icon" draggable="false" />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="no-images">No images available</div>
            )}
          </div>

          {/* Text Container - CSV Editor */}
          <div className="text-container">
            <div className="csv-display">
              <div className="csv-controls">
                <h3>Grading Sheet Editor</h3>
                <div className="action-buttons">
                  {!isEditing ? (
                    <>
                      <button className="btn btn-primary btn-small" onClick={() => setIsEditing(true)}>
                        EDIT SHEET
                      </button>
                      <button className="btn btn-success btn-small" onClick={handleRegradeClick} disabled={isRegrade}>
                        {isRegrade ? 'REGRADING...' : 'REGRADE EXAMS'}
                      </button>
                    </>
                  ) : (
                    <>
                      <button className="btn btn-success btn-small" onClick={saveChanges}>
                        Save Changes
                      </button>
                      <button
                        className="btn btn-secondary btn-small"
                        onClick={() => {
                          setIsEditing(false);
                          parseCsvData(csvData);
                        }}
                      >
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Hidden file input for answer key */}
              <input
                type="file"
                accept=".csv"
                onChange={handleAnswerKeyFileChange}
                ref={answerKeyInputRef}
                style={{ display: 'none' }}
              />

              {/* Regrade Options Modal */}
              {showRegradeOptions && (
                <div className="modal-overlay">
                  <div className="modal-content">
                    <h3>Regrade Options</h3>
                    <p>You have uploaded an answer key before. How would you like to proceed?</p>

                    <div className="regrade-options">
                      <button className="btn btn-primary btn-large" onClick={handleRegradeWithExistingKey}>
                        <i className="fas fa-recycle"></i>
                        Use Previous Answer Key
                        <small>Continue with: {previousAnswerKeyFileName}</small>
                      </button>

                      <button className="btn btn-info btn-large" onClick={handleRegradeWithNewKey}>
                        <i className="fas fa-upload"></i>
                        Upload New Answer Key
                        <small>Upload a different CSV answer key file</small>
                      </button>
                    </div>

                    <div className="modal-buttons">
                      <button className="btn btn-secondary" onClick={handleRegradeOptionsCancel}>
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Re-grading Progress Indicator */}
              {isRegrade && (
                <div className="regrade-progress">
                  <div className="progress-content">
                    <div className="spinner"></div>
                    <span>Regrading in progress... Please wait.</span>
                  </div>
                </div>
              )}

              <div className="csv-table-container">{renderCsvTable()}</div>
              {isEditing && (
                <div className="editing-instructions">
                  <p>Navigation: Use arrow keys to move between cells. Press Enter to move down.</p>
                  <p>Editing: Answer cells are auto-converted to uppercase. Invalid answers are highlighted.</p>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="no-results-message">
          <p>No data available. Please extract an exam first.</p>
          <Link to="/extract">
            <button className="btn btn-primary btn-large">Go to Extraction Page</button>
          </Link>
        </div>
      )}
    </div>
  );
}
export default SheetPage;
