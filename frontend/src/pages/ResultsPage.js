import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './ResultsPage.css';
import CustomAlert from '../components/CustomAlert';

import UniversityLogo from '../assets/logos/logo_hust.png';
import FamiLogo from '../assets/logos/logo_fami.png';

import NextIcon from '../assets/icons/next.png';
import PreviousIcon from '../assets/icons/previous.png';

import BackArrowIcon from '../assets/icons/back-arrow.png';
import TableIcon from '../assets/icons/table.png';
import DownloadIcon from '../assets/icons/download.png';
import DeleteIcon from '../assets/icons/delete.png';

import CheckIcon from '../assets/icons/check.png';

function ResultsPage() {
  const [csvData, setCsvData] = useState(null);
  const [images, setImages] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [hasResults, setHasResults] = useState(false);

  const [editedMetadata, setEditedMetadata] = useState({});
  const [editedAnswers, setEditedAnswers] = useState({});

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

  const handleClearResults = () => {
    showAlert('Do you want to clear all results data? This action cannot be undone.', 'warning', true, () => {
      localStorage.removeItem('examarkJobId');
      localStorage.removeItem('examarkCsvData');
      localStorage.removeItem('examarkImages');
      localStorage.removeItem('examarkEdits');
      setHasResults(false);
      closeAlert();
    });
  };

  // [MinIO] Use effect to load data from localStorage or URL parameters
  useEffect(() => {
    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const urlJobId = urlParams.get('jobId');

    // Get data from localStorage
    const savedJobId = localStorage.getItem('examarkJobId') || urlJobId;
    const savedCsvData = localStorage.getItem('examarkCsvData');
    const savedImages = localStorage.getItem('examarkImages');

    if (savedJobId && savedCsvData && savedImages) {
      setJobId(savedJobId);
      setCsvData(savedCsvData);

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

      // Load any saved edits
      const savedEdits = localStorage.getItem('examarkEdits');
      if (savedEdits) {
        const edits = JSON.parse(savedEdits);
        if (edits.metadata) {
          setEditedMetadata(edits.metadata);
        }
        if (edits.answers) {
          setEditedAnswers(edits.answers);
        }
      }
    }
  }, []);

  // Cập nhật function prepareEditedCsvData để return CSV thay vì chỉ prepare
  const updateCsvDataFromEdits = () => {
    if (!csvData) return null;

    const rows = csvData.split('\n');
    const updatedRows = [...rows];

    // Update metadata rows
    Object.keys(editedMetadata).forEach((imageIndex) => {
      const metadata = editedMetadata[imageIndex];
      const imageColumnIndex = parseInt(imageIndex) + 2;

      if (metadata['Student ID'] && updatedRows[1]) {
        const studentIdRow = updatedRows[1].split(',');
        if (studentIdRow.length > imageColumnIndex) {
          studentIdRow[imageColumnIndex] = metadata['Student ID'];
          updatedRows[1] = studentIdRow.join(',');
        }
      }

      if (metadata['Exam ID'] && updatedRows[2]) {
        const examIdRow = updatedRows[2].split(',');
        if (examIdRow.length > imageColumnIndex) {
          examIdRow[imageColumnIndex] = metadata['Exam ID'];
          updatedRows[2] = examIdRow.join(',');
        }
      }
    });

    // Update answer rows
    Object.keys(editedAnswers).forEach((key) => {
      const [imageIndex, part, questionInfo] = key.split('-');
      const imageColumnIndex = parseInt(imageIndex) + 2;
      const newValue = editedAnswers[key];

      let questionHeaderRow = 3;
      for (let i = 3; i < updatedRows.length; i++) {
        const cells = updatedRows[i].split(',');
        if (cells.length > 1 && cells[0].trim() === 'Part' && cells[1].trim() === 'Question') {
          questionHeaderRow = i;
          break;
        }
      }

      for (let i = questionHeaderRow + 1; i < updatedRows.length; i++) {
        const row = updatedRows[i].split(',');
        if (row.length > imageColumnIndex) {
          const rowPart = row[0].trim();
          const rowQuestion = row[1].trim();

          if (part === '1' && rowPart === '1') {
            const flatIndex = parseInt(questionInfo);
            const expectedQuestion = (flatIndex + 1).toString();
            if (rowQuestion === expectedQuestion) {
              row[imageColumnIndex] = newValue;
              updatedRows[i] = row.join(',');
              break;
            }
          } else if (part === '2' && rowPart === '2') {
            const [qIndex, rowIndex] = questionInfo.split('-');
            const questionNum = parseInt(qIndex) + 1;
            if (rowQuestion === questionNum.toString()) {
              let currentAnswer = row[imageColumnIndex] || '';
              const charIndex = parseInt(rowIndex);

              while (currentAnswer.length <= charIndex) {
                currentAnswer += 'X';
              }

              const answerArray = currentAnswer.split('');
              answerArray[charIndex] = newValue;
              row[imageColumnIndex] = answerArray.join('');
              updatedRows[i] = row.join(',');
              break;
            }
          }
        }
      }
    });

    const newCsvData = updatedRows.join('\n');

    localStorage.setItem('examarkCsvData', newCsvData);
    setCsvData(newCsvData);

    return newCsvData;
  };

  // Handle metadata edit using the current image index
  const handleMetadataEdit = (label, newValue) => {
    const updatedMetadata = {
      ...editedMetadata,
      [currentImageIndex]: {
        ...editedMetadata[currentImageIndex],
        [label]: newValue,
      },
    };

    setEditedMetadata(updatedMetadata);

    // Save to localStorage
    const editsToSave = {
      metadata: updatedMetadata,
      answers: editedAnswers,
    };
    localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));
    updateCsvDataFromEdits();
  };

  // Modify handleAnswerEdit to save to localStorage
  const handleAnswerEdit = (part, questionIdx, newValue) => {
    const key = `${currentImageIndex}-${part}-${questionIdx}`;
    const updatedAnswers = {
      ...editedAnswers,
      [key]: newValue.toUpperCase(),
    };

    setEditedAnswers(updatedAnswers);

    // Save to localStorage
    const editsToSave = {
      metadata: editedMetadata,
      answers: updatedAnswers,
    };
    localStorage.setItem('examarkEdits', JSON.stringify(editsToSave));
    updateCsvDataFromEdits();
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
    if (!csvData || images.length === 0) return { metadata: [], part1: [], part2: [], scoring: [] };

    // Parse CSV
    const rows = csvData.split('\n');
    if (rows.length < 4) return { metadata: [], part1: [], part2: [], scoring: [] };

    // Get the current image filename - NOW HANDLING OBJECT FORMAT
    const currentImageObj = images[currentImageIndex];
    let currentImageName;

    if (typeof currentImageObj === 'string') {
      // Old format - just a filename string
      currentImageName = currentImageObj;
    } else {
      // New format - object with name and url properties
      currentImageName = currentImageObj.name;
    }

    console.log('Current image name:', currentImageName);
    console.log('Current image object:', currentImageObj);

    const baseImageName = currentImageName.split('.')[0];

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

    console.log('Base image name:', baseImageName);
    console.log('Image column index:', imageColumnIndex);
    console.log('Header row:', headerRow);

    // If still not found, return empty
    if (imageColumnIndex === -1 || imageColumnIndex >= headerRow.length) {
      console.warn('Could not find column for image:', baseImageName);
      return { metadata: [], part1: [], part2: [], scoring: [] };
    }

    // Prepare result containers
    const metadata = [];
    const part1 = [];
    const part2 = [];
    const scoring = [];

    // Add metadata (Student ID, Exam ID)
    if (rows.length >= 3) {
      // Student ID row
      const studentIdRow = rows[1].split(',');
      if (studentIdRow.length > imageColumnIndex) {
        metadata.push({
          label: 'Student ID',
          value: studentIdRow[imageColumnIndex].trim(),
        });
      }

      // Exam ID row
      const examIdRow = rows[2].split(',');
      if (examIdRow.length > imageColumnIndex) {
        metadata.push({
          label: 'Exam ID',
          value: examIdRow[imageColumnIndex].trim(),
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

    // Process rows and separate answers from scoring
    const scoringLabels = ['Part 1 Correct', 'Part 2 Correct', 'Total Points'];

    for (let i = questionHeaderRow + 1; i < rows.length; i++) {
      const row = rows[i].split(',');
      if (row.length > imageColumnIndex) {
        const part = row[0].trim();
        const question = row[1].trim();
        const value = row[imageColumnIndex].trim();

        console.log(`Row ${i}: Part="${part}", Question="${question}", Value="${value}"`);

        // Check for scoring rows - NEW LOGIC HERE
        if (part === 'Part 1' && question === 'Correct') {
          console.log('Found Part 1 Correct score:', value);
          scoring.push({
            label: 'Part 1 Correct',
            value: value,
          });
        } else if (part === 'Part 2' && question === 'Correct') {
          console.log('Found Part 2 Correct score:', value);
          scoring.push({
            label: 'Part 2 Correct',
            value: value,
          });
        } else if (part === 'Total' && question === 'Points') {
          console.log('Found Total Points score:', value);
          scoring.push({
            label: 'Total Points',
            value: value,
          });
        } else if (part && question && !isNaN(question)) {
          // Regular answer row
          const item = {
            question: question,
            answer: value,
          };

          if (part === '1') {
            part1.push(item);
          } else if (part === '2') {
            part2.push(item);
          }
        }
      }
    }

    console.log('Final scoring array:', scoring);
    console.log('Part 1 questions:', part1.length);
    console.log('Part 2 questions:', part2.length);

    return { metadata, part1, part2, scoring };
  };

  // Render the answer with editable field
  const renderAnswer = (part, questionIdx, answer, cellId) => {
    const currentAnswer = getAnswerValue(part, questionIdx, answer);
    let style = {};

    const allowedAnswers = part === '1' ? ['A', 'B', 'C', 'D', '_'] : ['D', 'S', '_'];

    if (answer && answer.match(/[a-z]/)) {
      style = { backgroundColor: 'cyan' };
    } else if (part === '2' && currentAnswer && currentAnswer.includes('_')) {
      style = { backgroundColor: '#d0f5dd' };
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
          numCols = 6;
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
    if (cellId) {
      cellProps.id = cellId;
      cellProps.onKeyDown = onKeyDownHandler;
      cellProps.onInput = onInputHandler;
    }

    return (
      <span
        contentEditable={true}
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

  // Add this function to prepare edited CSV data
  const prepareEditedCsvData = () => {
    if (!csvData) return null;

    const rows = csvData.split('\n');
    const updatedRows = [...rows];

    // Update metadata rows (Student ID and Exam ID)
    Object.keys(editedMetadata).forEach((imageIndex) => {
      const metadata = editedMetadata[imageIndex];
      const imageColumnIndex = parseInt(imageIndex) + 2; // Assuming page_0 is at column index 2

      if (metadata['Student ID'] && updatedRows[1]) {
        const studentIdRow = updatedRows[1].split(',');
        if (studentIdRow.length > imageColumnIndex) {
          studentIdRow[imageColumnIndex] = metadata['Student ID'];
          updatedRows[1] = studentIdRow.join(',');
        }
      }

      if (metadata['Exam ID'] && updatedRows[2]) {
        const examIdRow = updatedRows[2].split(',');
        if (examIdRow.length > imageColumnIndex) {
          examIdRow[imageColumnIndex] = metadata['Exam ID'];
          updatedRows[2] = examIdRow.join(',');
        }
      }
    });

    // Update answer rows
    Object.keys(editedAnswers).forEach((key) => {
      const [imageIndex, part, questionInfo] = key.split('-');
      const imageColumnIndex = parseInt(imageIndex) + 2;
      const newValue = editedAnswers[key];

      // Find the corresponding row in CSV
      let questionHeaderRow = 3;
      for (let i = 3; i < updatedRows.length; i++) {
        const cells = updatedRows[i].split(',');
        if (cells.length > 1 && cells[0].trim() === 'Part' && cells[1].trim() === 'Question') {
          questionHeaderRow = i;
          break;
        }
      }

      // Update the specific answer
      for (let i = questionHeaderRow + 1; i < updatedRows.length; i++) {
        const row = updatedRows[i].split(',');
        if (row.length > imageColumnIndex) {
          const rowPart = row[0].trim();
          const rowQuestion = row[1].trim();

          if (part === '1' && rowPart === '1') {
            const flatIndex = parseInt(questionInfo);
            const expectedQuestion = (flatIndex + 1).toString();
            if (rowQuestion === expectedQuestion) {
              row[imageColumnIndex] = newValue;
              updatedRows[i] = row.join(',');
              break;
            }
          } else if (part === '2' && rowPart === '2') {
            const [qIndex, rowIndex] = questionInfo.split('-');
            // For Part 2, we need to update the character at the specific position
            const questionNum = parseInt(qIndex) + 1;
            if (rowQuestion === questionNum.toString()) {
              let currentAnswer = row[imageColumnIndex] || '';
              const charIndex = parseInt(rowIndex);

              // Ensure the string is long enough
              while (currentAnswer.length <= charIndex) {
                currentAnswer += 'X';
              }

              // Replace the character at the specific position
              const answerArray = currentAnswer.split('');
              answerArray[charIndex] = newValue;
              row[imageColumnIndex] = answerArray.join('');
              updatedRows[i] = row.join(',');
              break;
            }
          }
        }
      }
    });

    return updatedRows.join('\n');
  };

  // Handle saving edited data as Excel
  const handleSaveExcel = () => {
    try {
      const editedCsv = prepareEditedCsvData();

      if (!editedCsv) {
        alert('No data to save');
        return;
      }

      // Parse CSV data into rows
      const csvRows = editedCsv.split('\n').filter((row) => row.trim());
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

      // Create blob with Excel MIME type
      const blob = new Blob([excelXML], {
        type: 'application/vnd.ms-excel',
      });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `grading_results_${jobId}.xls`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error saving Excel:', error);
      alert('Error saving Excel file');
    }
  };

  const isValidId = (value) => {
    return /^\d*$/.test(value.toString().trim());
  };

  return (
    <div className="ResultsPage">
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

      {hasResults ? (
        <>
          {/* Header Section */}
          <header className="page-header">
            <div className="page-header-left">
              <img src={UniversityLogo} alt="HUST Logo" className="page-header-logo" draggable="false" />
            </div>
            <div className="page-header-center">
              <h1>Student Exam Detail</h1>
              <p>Review and edit your graded exams one by one</p>
            </div>
            <div className="page-header-right">
              <div className="header-buttons">
                <Link to="/" draggable="false">
                  <button className="header-btn header-btn-primary">
                    Dashboard
                    <img src={BackArrowIcon} alt="Back" className="header-btn-icon" draggable="false" />
                  </button>
                </Link>
                <Link to="/sheet" draggable="false">
                  <button className="header-btn header-btn-primary">
                    Review Sheet
                    <img src={TableIcon} alt="Table" className="header-btn-icon" draggable="false" />
                  </button>
                </Link>
                <button className="header-btn header-btn-secondary" onClick={handleSaveExcel}>
                  Save Excel
                  <img src={DownloadIcon} alt="Download" className="header-btn-icon" draggable="false" />
                </button>
                <button className="header-btn header-btn-danger" onClick={handleClearResults}>
                  Clear Results
                  <img src={DeleteIcon} alt="Delete" className="header-btn-icon" draggable="false" />
                </button>
              </div>
              <img src={FamiLogo} alt="Fami Logo" className="page-header-fami-logo" draggable="false" />
            </div>
          </header>

          {/* Image Container */}
          <div className="image-container">
            {images.length > 0 ? (
              <>
                <img
                  //src={`http://127.0.0.1:8080/results/${jobId}/images/${images[currentImageIndex]}`}
                  src={images[currentImageIndex].url}
                  alt={`Graded exam page ${currentImageIndex + 1}`}
                  className="result-image"
                  onError={(e) => {
                    console.error('Failed to load image from MinIO:', e.target.src);
                    e.target.src =
                      'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y4ZjlmYSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2Yzc1N2QiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pjwvc3ZnPg==';
                  }}
                />
                <div className="results-image-navigation">
                  <div className="results-image-navigation-buttons">
                    <button className="results-nav-button" onClick={showPrevImage} disabled={currentImageIndex === 0}>
                      <img src={PreviousIcon} alt="Previous" className="nav-icon" />
                    </button>
                    <span className="results-nav-text">
                      Exam {} {currentImageIndex + 1} of {images.length} - Page{' '}
                      {images[currentImageIndex]?.name.match(/page_(\d+)/)?.[1] || currentImageIndex + 1}
                    </span>
                    <button
                      className="results-nav-button"
                      onClick={showNextImage}
                      disabled={currentImageIndex === images.length - 1}
                    >
                      <img src={NextIcon} alt="Next" className="nav-icon" />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="no-images">No images available</div>
            )}
          </div>

          {/* Text Container */}
          <div className="text-container">
            {(() => {
              const results = getCurrentImageResults();
              return (
                <>
                  {/* Combined Metadata and Scoring Section */}
                  <div className="scoring-section">
                    <div className="scoring-header">
                      <strong>Exam Summary</strong>
                    </div>
                    <div className="scoring-items">
                      {/* Student ID and Exam ID */}
                      {results.metadata.map((item, index) => (
                        <div className="scoring-item" key={`meta-${index}`}>
                          <span className="scoring-label">{item.label}</span>
                          {item.label === 'Student ID' || item.label === 'Exam ID' ? (
                            <span
                              className="scoring-value editable"
                              contentEditable
                              suppressContentEditableWarning
                              style={{
                                backgroundColor: !isValidId(
                                  (editedMetadata[currentImageIndex] &&
                                    editedMetadata[currentImageIndex][item.label]) ||
                                    item.value,
                                )
                                  ? '#ffeb3b'
                                  : 'white',
                              }}
                              onInput={(e) => {
                                // Only allow numbers
                                const value = e.target.textContent.replace(/\D/g, '');
                                if (value !== e.target.textContent) {
                                  e.target.textContent = value;
                                  // Move cursor to end
                                  const range = document.createRange();
                                  const sel = window.getSelection();
                                  range.selectNodeContents(e.target);
                                  range.collapse(false);
                                  sel.removeAllRanges();
                                  sel.addRange(range);
                                }
                              }}
                              onBlur={(e) => handleMetadataEdit(item.label, e.target.textContent.trim())}
                            >
                              {(editedMetadata[currentImageIndex] && editedMetadata[currentImageIndex][item.label]) ||
                                item.value}
                            </span>
                          ) : (
                            <span className="scoring-value">{item.value}</span>
                          )}
                        </div>
                      ))}

                      {/* Scoring Information */}
                      {results.scoring.map((item, index) => (
                        <div className="scoring-item" key={`score-${index}`}>
                          <span className="scoring-label">{item.label}</span>
                          <span className="scoring-value">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Part 1 */}
                  <div className="part-label">
                    <strong> Content Part 1</strong>
                  </div>
                  <div className="part1-grid">
                    {[0, 1, 2].map((colIndex) => (
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
                    <strong>Content Part 2</strong>
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
                                item.answer && item.answer.length > rowIndex ? item.answer[rowIndex] : 'X';
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
                </>
              );
            })()}
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
