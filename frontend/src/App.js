import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to connect to API. Make sure Flask server is running!');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    if (status === 'success') return '#22c55e';
    if (status === 'warning') return '#f59e0b';
    if (status === 'danger') return '#ef4444';
    return '#6b7280';
  };

  const getStatusBg = (status) => {
    if (status === 'success') return '#f0fdf4';
    if (status === 'warning') return '#fffbeb';
    if (status === 'danger') return '#fef2f2';
    return '#f9fafb';
  };

  const getStatusEmoji = (status) => {
    if (status === 'success') return '✅';
    if (status === 'warning') return '⚠️';
    if (status === 'danger') return '🚨';
    return '🔍';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">☀️</div>
          <div>
            <h1>Solar Panel Fault Detection</h1>
            <p>AI-Powered Fault Analysis System</p>
          </div>
        </div>
      </header>

      <main className="main">
        {/* Upload Section */}
        <div className="card upload-card">
          <h2>Upload Solar Panel Image</h2>
          <p className="subtitle">Upload an image to detect faults using our CNN model</p>

          <div
            className="upload-area"
            onClick={() => document.getElementById('fileInput').click()}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="preview-img" />
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📁</div>
                <p>Click to upload image</p>
                <span>Supports JPG, PNG</span>
              </div>
            )}
          </div>

          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: 'none' }}
          />

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={!selectedImage || loading}
          >
            {loading ? '🔄 Analyzing...' : '🔍 Detect Fault'}
          </button>

          {error && <div className="error-box">{error}</div>}
        </div>

        {/* Result Section */}
        {result && (
          <div className="card result-card" style={{ background: getStatusBg(result.status) }}>
            <h2>Detection Result</h2>

            <div className="result-main" style={{ borderColor: getStatusColor(result.status) }}>
              <div className="result-emoji">{getStatusEmoji(result.status)}</div>
              <div className="result-info">
                <h3 style={{ color: getStatusColor(result.status) }}>
                  {result.predicted_class}
                </h3>
                <p className="confidence">Confidence: <strong>{result.confidence}%</strong></p>
                <p className="result-message">{result.message}</p>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="confidence-bar-container">
              <div
                className="confidence-bar"
                style={{
                  width: `${result.confidence}%`,
                  background: getStatusColor(result.status)
                }}
              />
            </div>

            {/* All Probabilities */}
            <h3 className="prob-title">All Class Probabilities</h3>
            <div className="probabilities">
              {Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => (
                  <div key={cls} className="prob-item">
                    <span className="prob-label">{cls}</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar"
                        style={{
                          width: `${prob}%`,
                          background: cls === result.predicted_class
                            ? getStatusColor(result.status)
                            : '#cbd5e1'
                        }}
                      />
                    </div>
                    <span className="prob-value">{prob}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Info Cards */}
        <div className="info-grid">
          {[
            { icon: '✅', label: 'Clean', desc: 'Normal operation' },
            { icon: '🌫️', label: 'Dusty', desc: 'Needs cleaning' },
            { icon: '🐦', label: 'Bird-drop', desc: 'Spot cleaning needed' },
            { icon: '⚡', label: 'Electrical', desc: 'Urgent inspection' },
            { icon: '💥', label: 'Physical', desc: 'Repair needed' },
            { icon: '❄️', label: 'Snow', desc: 'Remove snow' },
          ].map((item) => (
            <div key={item.label} className="info-card">
              <div className="info-icon">{item.icon}</div>
              <div className="info-label">{item.label}</div>
              <div className="info-desc">{item.desc}</div>
            </div>
          ))}
        </div>
      </main>

      <footer className="footer">
        <p>Solar Panel Fault Detection System | CNN Model (MobileNetV2) | Accuracy: 83.33%</p>
      </footer>
    </div>
  );
}

export default App;