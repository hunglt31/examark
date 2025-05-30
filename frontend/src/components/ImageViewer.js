import React from 'react';
import './ImageViewer.css';

const ImageViewer = ({ images, currentImageIndex, onImageChange }) => {
  const handleNextImage = () => {
    if (currentImageIndex < images.length - 1) {
      onImageChange(currentImageIndex + 1);
    }
  };

  const handlePrevImage = () => {
    if (currentImageIndex > 0) {
      onImageChange(currentImageIndex - 1);
    }
  };

  return (
    <div className="image-viewer">
      <img 
        src={images[currentImageIndex]} 
        alt={`Image ${currentImageIndex + 1}`} 
        className="result-image" 
      />
      <div className="image-navigation">
        <button onClick={handlePrevImage} disabled={currentImageIndex === 0}>
          Previous
        </button>
        <span>Image {currentImageIndex + 1} of {images.length}</span>
        <button onClick={handleNextImage} disabled={currentImageIndex === images.length - 1}>
          Next
        </button>
      </div>
    </div>
  );
};

export default ImageViewer;