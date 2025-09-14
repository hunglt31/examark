import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './ReviewPage.css';
import CustomAlert from '../components/CustomAlert';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

import BackArrowIcon from '../assets/icons/back-arrow.png';
import ResultIcon from '../assets/icons/result.png';
import DownloadIcon from '../assets/icons/download.png';
import DeleteIcon from '../assets/icons/delete.png';

import NextIcon from '../assets/icons/next.png';
import PreviousIcon from '../assets/icons/previous.png';

function ReviewPage() {
  const [examData, setExamData] = useState(null);
  const [csvData, setCsvData] = useState('');
  const [csvRows, setCsvRows] = useState([]);
  const [images, setImages] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [hasResults, setHasResults] = useState(false);
  const [selectedCell, setSelectedCell] = useState({ row: 0, col: 0 });
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const answerKeyInputRef = useRef(null);
  const tableRef = useRef(null);

  // Helper function to check if a cell is invalid
  const isCellInvalid = (value, rowIndex, colIndex) => {
    if (!value || value.trim() === '') return false;

    if (rowIndex < 4) {
      // Header rows - check for invalid IDs
      if ((rowIndex === 1 || rowIndex === 2) && colIndex > 1) {
        // Invalid if Student ID or Exam ID contains "_" or non-numeric characters
        const stringValue = value.toString().trim();
        return stringValue.includes('_') || !/^\d+$/.test(stringValue);
      }
      return false;
    }

    // Answer rows
    if (colIndex < 2) return false;

    const part = csvRows[rowIndex]?.[0] || '';
    const stringValue = value.toString().trim();

    // Check for lowercase letters (always invalid)
    if (/[a-z]/.test(stringValue)) return true;

    // Check for invalid answers based on part
    if (part === '1') {
      const validAnswers = ['A', 'B', 'C', 'D', '_'];
      return !validAnswers.includes(stringValue.toUpperCase());
    } else if (part === '2') {
      // Invalid if contains 'x' or 'X', or doesn't match valid pattern
      if (/[xX]/.test(stringValue)) return true;
      const validPattern = /^[DS_]+$/i;
      return !validPattern.test(stringValue);
    }

    return false;
  };

  // Check if an exam (column) has any invalid cells
  const examHasInvalidCells = (colIndex) => {
    if (colIndex < 2 || !csvRows.length) return false;

    for (let rowIndex = 0; rowIndex < csvRows.length; rowIndex++) {
      const row = csvRows[rowIndex];
      if (row && row[colIndex] !== undefined && row[colIndex] !== '') {
        if (isCellInvalid(row[colIndex], rowIndex, colIndex)) {
          console.log(`Invalid cell found at row ${rowIndex}, col ${colIndex}:`, row[colIndex]);
          return true;
        }
      }
    }
    return false;
  };

  // Load and combine all CSV data from all PDFs
  const loadAllCsvData = async (examData) => {
    setLoading(true);
    let combinedRows = [];
    let combinedImages = [];
    let examToImageMap = [];

    try {
      for (let pdfIndex = 0; pdfIndex < examData.data.length; pdfIndex++) {
        const pdfResult = examData.data[pdfIndex];
        const className = pdfResult.pdf || `Class ${pdfIndex + 1}`;

        console.log(`Processing PDF ${pdfIndex}:`, className);

        // Get CSV data
        let csvDataToUse = pdfResult.csvData || pdfResult.csv;
        let csvText = '';

        if (csvDataToUse && csvDataToUse.startsWith('http')) {
          try {
            const response = await fetch(csvDataToUse);
            csvText = await response.text();
          } catch (error) {
            console.error('Error fetching CSV data for', className, error);
            continue;
          }
        } else {
          csvText = csvDataToUse;
        }

        // Parse CSV rows
        const rows = csvText.split('\n').map((line) => line.split(',').map((cell) => cell.trim()));

        if (rows.length === 0) continue;

        // Process images for this PDF
        let pdfImages = [];
        if (pdfResult.images) {
          if (Array.isArray(pdfResult.images)) {
            pdfImages = pdfResult.images;
          } else if (typeof pdfResult.images === 'object') {
            pdfImages = Object.entries(pdfResult.images).map(([name, url]) => ({
              name,
              url,
              className,
            }));
          }
        }

        // Sort images by page number
        pdfImages.sort((a, b) => {
          const pageNumA = parseInt(a.name?.match(/page_(\d+)/)?.[1] || '0', 10);
          const pageNumB = parseInt(b.name?.match(/page_(\d+)/)?.[1] || '0', 10);
          return pageNumA - pageNumB;
        });

        // Add class name to images
        const imagesWithClass = pdfImages.map((img) => ({
          ...img,
          className,
          displayName: `${className} - ${img.name}`,
        }));

        if (pdfIndex === 0) {
          combinedRows = [...rows];
          for (let col = 2; col < rows[0]?.length; col++) {
            const imageIndex = col - 2;
            if (imageIndex < imagesWithClass.length) {
              examToImageMap.push({
                columnIndex: col,
                imageIndex: combinedImages.length + imageIndex,
                className,
                imageName: imagesWithClass[imageIndex]?.name || `page_${imageIndex + 1}`,
              });
            }
          }
        } else {
          const dataRows = rows.slice(4);
          for (let headerRowIndex = 0; headerRowIndex < 4; headerRowIndex++) {
            if (combinedRows[headerRowIndex]) {
              const newColumns = rows[headerRowIndex]?.slice(2) || [];
              combinedRows[headerRowIndex] = combinedRows[headerRowIndex].concat(newColumns);
            }
          }
          for (let dataRowIndex = 0; dataRowIndex < dataRows.length; dataRowIndex++) {
            const combinedRowIndex = dataRowIndex + 4;
            if (combinedRowIndex < combinedRows.length) {
              const newColumns = dataRows[dataRowIndex]?.slice(2) || [];
              combinedRows[combinedRowIndex] = combinedRows[combinedRowIndex].concat(newColumns);
            }
          }
          const startingColumn = combinedRows[0].length - (rows[0]?.length - 2 || 0);
          for (let col = 2; col < rows[0]?.length; col++) {
            const imageIndex = col - 2;
            const combinedCol = startingColumn + imageIndex;
            if (imageIndex < imagesWithClass.length) {
              examToImageMap.push({
                columnIndex: combinedCol,
                imageIndex: combinedImages.length + imageIndex,
                className,
                imageName: imagesWithClass[imageIndex]?.name || `page_${imageIndex + 1}`,
              });
            }
          }
        }

        combinedImages = combinedImages.concat(imagesWithClass);
      }

      // Create a helper function that works with the current data
      const checkCellInvalid = (value, rowIndex, colIndex, rows) => {
        if (!value || value.trim() === '') return false;

        if (rowIndex < 4) {
          if ((rowIndex === 1 || rowIndex === 2) && colIndex > 1) {
            const stringValue = value.toString().trim();
            const isInvalid = stringValue.includes('_') || !/^\d+$/.test(stringValue);
            if (isInvalid) {
              console.log(`Invalid ID found at row ${rowIndex}, col ${colIndex}: "${stringValue}"`);
            }
            return isInvalid;
          }
          return false;
        }

        if (colIndex < 2) return false;

        const part = rows[rowIndex]?.[0] || '';
        const stringValue = value.toString().trim();

        if (/[a-z]/.test(stringValue)) return true;

        if (part === '1') {
          const validAnswers = ['A', 'B', 'C', 'D', '_'];
          return !validAnswers.includes(stringValue.toUpperCase());
        } else if (part === '2') {
          if (/[xX]/.test(stringValue)) return true;
          const validPattern = /^[DS_]+$/i;
          return !validPattern.test(stringValue);
        }

        return false;
      };

      // Check which exams have invalid cells using the combined data
      const validColumnIndices = new Set([0, 1]);
      const validExamToImageMap = [];

      examToImageMap.forEach((mapping) => {
        let hasInvalid = false;

        // Check all rows for this column
        for (let rowIndex = 0; rowIndex < combinedRows.length; rowIndex++) {
          const row = combinedRows[rowIndex];
          if (row && row[mapping.columnIndex] !== undefined && row[mapping.columnIndex] !== '') {
            if (checkCellInvalid(row[mapping.columnIndex], rowIndex, mapping.columnIndex, combinedRows)) {
              hasInvalid = true;
              console.log(
                `Invalid cell found in column ${mapping.columnIndex} (${mapping.className}):`,
                `Row ${rowIndex}, Value: "${row[mapping.columnIndex]}"`,
              );
              break;
            }
          }
        }

        console.log(`Column ${mapping.columnIndex} (${mapping.className}): Has invalid cells: ${hasInvalid}`);

        if (hasInvalid) {
          validColumnIndices.add(mapping.columnIndex);
          validExamToImageMap.push(mapping);
        }
      });

      console.log('Valid columns to show:', Array.from(validColumnIndices));
      console.log('Valid exam mappings:', validExamToImageMap);

      if (validExamToImageMap.length === 0) {
        setHasResults(false);
        setLoading(false);
        return;
      }

      // Filter combined rows to only include valid columns
      const filteredRows = combinedRows.map((row) => row.filter((cell, index) => validColumnIndices.has(index)));

      // Update the mapping indices after filtering
      const sortedValidColumns = Array.from(validColumnIndices).sort((a, b) => a - b);
      const validExamToImageMapFiltered = validExamToImageMap.map((mapping) => ({
        ...mapping,
        columnIndex: sortedValidColumns.indexOf(mapping.columnIndex),
      }));

      // Set the state with filtered data
      setCsvData(filteredRows.map((row) => row.join(',')).join('\n'));
      setCsvRows(filteredRows);
      setImages(combinedImages);
      setJobId('combined_review');
      setHasResults(filteredRows.length > 0 && validExamToImageMapFiltered.length > 0);

      // Store the exam to image mapping
      window.examToImageMap = validExamToImageMapFiltered;
    } catch (error) {
      console.error('Error loading CSV data:', error);
      setHasResults(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const storedExamData = localStorage.getItem('examData');
    if (storedExamData) {
      const parsedExamData = JSON.parse(storedExamData);
      setExamData(parsedExamData);
      if (parsedExamData.data && parsedExamData.data.length > 0) {
        loadAllCsvData(parsedExamData);
      }
    }
  }, []);

  // Alert state and functions
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

  // Handle cell click and map to corresponding image
  const handleCellClick = (rowIndex, colIndex) => {
    setSelectedCell({ row: rowIndex, col: colIndex });

    // Find corresponding image based on column
    if (window.examToImageMap && colIndex >= 2) {
      const mapping = window.examToImageMap.find((m) => m.columnIndex === colIndex);
      if (mapping && mapping.imageIndex < images.length) {
        setCurrentImageIndex(mapping.imageIndex);
      }
    }
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

    // Auto uppercase for answer cells
    if (rowIndex > 3 && colIndex > 1) {
      const part = csvRows[rowIndex]?.[0] || '';

      if (part === '2') {
        const validChars = value.replace(/[^A-Za-z]/g, '');
        const limitedValue = validChars.slice(0, 6).toUpperCase();
        if (limitedValue !== value) {
          e.target.textContent = limitedValue;
          const range = document.createRange();
          const sel = window.getSelection();
          range.selectNodeContents(e.target);
          range.collapse(false);
          sel.removeAllRanges();
          sel.addRange(range);
        }
      } else {
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

  // Determine cell style based on content
  const getCellStyle = (value, rowIndex, colIndex) => {
    if (isCellInvalid(value, rowIndex, colIndex)) {
      if (value && /[a-z]/.test(value)) {
        return { backgroundColor: 'cyan' };
      } else {
        return { backgroundColor: 'yellow' };
      }
    }
    return {};
  };

  // Navigation functions
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

  // Save changes function
  const saveChanges = async () => {
    const updatedCsv = updateCsvFromRows();
    setIsEditing(false);
    showAlert('Changes saved successfully!', 'success');
  };

  // Render CSV table
  const renderCsvTable = () => {
    if (!csvRows.length) return <div>No CSV data available</div>;

    const headerRows = csvRows.slice(0, 4);
    let bodyRows = csvRows.slice(4);

    // Filter out scoring rows
    bodyRows = bodyRows.filter((row) => {
      const rowLabel = row[1]?.toLowerCase() || '';
      return !rowLabel.includes('correct') && !rowLabel.includes('points');
    });

    if (csvRows[0].length <= 2) {
      return (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <h3>No exams with invalid results found.</h3>
          <p>All exam results appear to be valid.</p>
        </div>
      );
    }

    return (
      <table className="csv-table review-mode" ref={tableRef}>
        <thead className="csv-table-head">
          {headerRows.map((row, rowIndex) => (
            <tr key={`header-${rowIndex}`}>
              {row.map((cell, colIndex) => {
                const isSelected = selectedCell.row === rowIndex && selectedCell.col === colIndex;
                const isFrozenColumn = colIndex < 2;
                const isNonEditableRow = rowIndex === 0 || rowIndex === 3;
                const isEditable = isEditing && !isFrozenColumn && !isNonEditableRow;
                const hasIssues = colIndex >= 2 && examHasInvalidCells(colIndex);

                return (
                  <th key={colIndex} className={hasIssues ? 'has-issues' : ''}>
                    <div
                      id={`cell-${rowIndex}-${colIndex}`}
                      className={`csv-cell ${isEditable ? 'editable' : ''} ${
                        isSelected ? 'current-cell' : ''
                      } ${isFrozenColumn || isNonEditableRow ? 'column-frozen' : ''}`}
                      contentEditable={isEditable}
                      suppressContentEditableWarning={true}
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
          {bodyRows.map((row, index) => {
            const actualRowIndex = index + 4;
            return (
              <tr key={`body-${index}`}>
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
                        className={`csv-cell ${isEditable ? 'editable' : ''} ${
                          isSelected ? 'current-cell' : ''
                        } ${isFrozenColumn ? 'column-frozen' : ''}`}
                        contentEditable={isEditable}
                        suppressContentEditableWarning={true}
                        onInput={(e) => !isFrozenColumn && handleInput(e, actualRowIndex, colIndex)}
                        onBlur={(e) =>
                          !isFrozenColumn && handleCellEdit(actualRowIndex, colIndex, e.target.textContent)
                        }
                        onClick={() => handleCellClick(actualRowIndex, colIndex)}
                        tabIndex={isEditable ? 0 : -1}
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

  const handleSaveExcel = () => {
    try {
      const currentCsvData = updateCsvFromRows();
      if (!currentCsvData) {
        showAlert('No data to save', 'error');
        return;
      }

      const csvRows = currentCsvData.split('\n').filter((row) => row.trim());
      const data = csvRows.map((row) => row.split(',').map((cell) => cell.trim()));

      let excelXML = `<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
 <Worksheet ss:Name="Review Results">
  <Table>`;

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

      const blob = new Blob([excelXML], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `review_results_${new Date().getTime()}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error saving Excel:', error);
      showAlert('Error saving Excel file', 'error');
    }
  };

  return (
    <div className="ReviewPage">
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

      <header className="page-header">
        <div className="page-header-left">
          <img src={UniversityLogo} alt="HUST Logo" className="page-header-logo" draggable="false" />
        </div>
        <div className="page-header-center">
          <h1>Review Invalid Results</h1>
          <p>Edit only exams with invalid answers from all classes</p>
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

      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <h3>Loading exam data...</h3>
          <p>Processing all classes and filtering invalid results...</p>
        </div>
      ) : hasResults ? (
        <>
          <div className="image-container">
            {images.length > 0 ? (
              <>
                <div className="image-header">
                  <h3 className="class-name">{images[currentImageIndex]?.className || 'No Class'}</h3>
                  <p className="image-name">{images[currentImageIndex]?.name || 'No Image'}</p>
                </div>
                <img
                  src={images[currentImageIndex]?.url}
                  alt={`Exam page ${currentImageIndex + 1}`}
                  className="result-image"
                  onError={(e) => {
                    console.error('Image load error:', e.target.src);
                  }}
                />
                <div className="sheet-image-navigation">
                  <div className="sheet-image-navigation-buttons">
                    <button className="sheet-nav-button" onClick={showPrevImage} disabled={currentImageIndex === 0}>
                      <img src={PreviousIcon} alt="Previous" className="nav-icon" draggable="false" />
                    </button>
                    <span className="sheet-nav-text">
                      {images[currentImageIndex]?.displayName || `Image ${currentImageIndex + 1} of ${images.length}`}
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

          <div className="text-container">
            <div className="csv-display">
              <div className="csv-controls">
                <h3>Invalid Results Editor</h3>
                <div className="review-controls">
                  <div className="review-stats">
                    {csvRows.length > 0 && (
                      <span>Showing {Math.max(0, csvRows[0].length - 2)} exams with invalid answers</span>
                    )}
                  </div>
                </div>
                <div className="action-buttons">
                  {!isEditing ? (
                    <button className="btn btn-primary btn-small" onClick={() => setIsEditing(true)}>
                      EDIT SHEET
                    </button>
                  ) : (
                    <>
                      <button className="btn btn-success btn-small" onClick={saveChanges}>
                        Save Changes
                      </button>
                      <button
                        className="btn btn-secondary btn-small"
                        onClick={() => {
                          setIsEditing(false);
                          if (examData) loadAllCsvData(examData);
                        }}
                      >
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </div>

              <div className="csv-table-container">{renderCsvTable()}</div>

              {isEditing && (
                <div className="editing-instructions">
                  <p>Navigation: Click on cells to select and edit. Invalid answers are highlighted.</p>
                  <p>Color coding: Yellow = Invalid answers, Cyan = Lowercase letters</p>
                </div>
              )}

              {csvRows.length > 0 && csvRows[0].length > 2 && (
                <div className="review-legend">
                  <h4>Color Key:</h4>
                  <div className="legend-item">
                    <div className="legend-color invalid-answers"></div>
                    <span>Yellow: Invalid Cell</span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color lowercase"></div>
                    <span>Cyan: System Suggestion</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="no-results-message">
          <h3>No invalid results found</h3>
          <p>All exam results appear to be valid, or no exam data is available.</p>
          <Link to="/extract">
            <button className="btn btn-primary btn-large">Go to Extraction Page</button>
          </Link>
        </div>
      )}
    </div>
  );
}

export default ReviewPage;
