import React from 'react';
import './CustomAlert.css';

const CustomAlert = ({ 
  isOpen, 
  message, 
  type = 'info', 
  onClose, 
  showConfirm = false, 
  onConfirm, 
  confirmText = 'Confirm', 
  cancelText = 'Cancel' 
}) => {
  if (!isOpen) return null;

  const getIcon = () => {
    switch (type) {
      case 'success': return '✓';
      case 'error': return '✕';
      case 'warning': return '⚠';
      default: return 'ℹ';
    }
  };

  const handleConfirm = () => {
    if (onConfirm) {
      onConfirm();
    }
  };

  return (
    <div className="custom-alert-overlay">
      <div className={`custom-alert custom-alert-${type}`}>
        <div className="custom-alert-header">
          <div className="custom-alert-icon">{getIcon()}</div>
        </div>
        <div className="custom-alert-content">
          <div className="custom-alert-message">{message}</div>
          <div className="custom-alert-buttons">
            {showConfirm ? (
              <>
                <button className="custom-alert-button custom-alert-confirm" onClick={handleConfirm}>
                  {confirmText}
                </button>
                <button className="custom-alert-button custom-alert-cancel" onClick={onClose}>
                  {cancelText}
                </button>
              </>
            ) : (
              <button className="custom-alert-close" onClick={onClose}>
                OK
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomAlert;