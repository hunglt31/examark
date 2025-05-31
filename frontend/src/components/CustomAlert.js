import React from 'react';
import './CustomAlert.css';

const CustomAlert = ({ isOpen, message, type = 'info', onClose }) => {
  if (!isOpen) return null;

  const getIcon = () => {
    switch (type) {
      case 'success': return '✓';
      case 'error': return '✕';
      case 'warning': return '⚠';
      default: return 'ℹ';
    }
  };

  return (
    <div className="custom-alert-overlay">
      <div className={`custom-alert custom-alert-${type}`}>
        <div className="custom-alert-icon">{getIcon()}</div>
        <div className="custom-alert-message">{message}</div>
        <button className="custom-alert-close" onClick={onClose}>
          OK
        </button>
      </div>
    </div>
  );
};

export default CustomAlert;