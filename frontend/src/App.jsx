import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, CheckCircle, Download, Film, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = '';

function App() {
  const [library, setLibrary] = useState([]);
  const [libraryError, setLibraryError] = useState(false);
  const [selectedRef, setSelectedRef] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    const fetchLibrary = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/library`);
        if (!res.ok) throw new Error("Backend not ready");
        const data = await res.json();
        setLibrary(data);
        setLibraryError(false);
      } catch (err) {
        setLibraryError(true);
        setTimeout(fetchLibrary, 2000); // Retry every 2 seconds
      }
    };
    fetchLibrary();
  }, []);

  const handleFileDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setTargetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setTargetFile(e.target.files[0]);
    }
  };

  const handleGrade = async () => {
    if (!selectedRef || !targetFile) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      // We need to fetch the reference image as a File object to send it
      const refResponse = await fetch(`${API_BASE}${selectedRef.path}`);
      const refBlob = await refResponse.blob();
      const refFile = new File([refBlob], selectedRef.name + '.jpg', { type: 'image/jpeg' });

      const formData = new FormData();
      formData.append('reference', refFile);
      formData.append('target', targetFile);
      formData.append('steps', '25');
      formData.append('size', '512');
      formData.append('ncc', 'true');

      const response = await fetch(`${API_BASE}/api/grade`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const initData = await response.json();
      const taskId = initData.task_id;
      
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        const statusRes = await fetch(`${API_BASE}/api/status/${taskId}`);
        if (!statusRes.ok) throw new Error(await statusRes.text());
        
        const statusData = await statusRes.json();
        if (statusData.status === 'completed') {
          setResult(statusData.result);
          break;
        } else if (statusData.status === 'error') {
          throw new Error(statusData.error || "Unknown processing error");
        }
      }
    } catch (err) {
      console.error("Grading error:", err);
      setError(`Failed: ${err.message}. ${err.cause ? 'Cause: ' + err.cause : ''}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="hero">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <h1>CineGrade AI</h1>
          <p>The industry-standard neural color grading engine. Transform your RAW footage into cinematic masterpieces instantly with zero quality loss.</p>
        </motion.div>
      </header>

      {!result && (
        <div className="layout-grid">
          <motion.div className="glass-panel" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
            <h3>1. Select Cinematic Look</h3>
            {library.length === 0 ? (
              <div style={{ padding: '60px 20px', textAlign: 'center', color: '#a0a0a0', backgroundColor: 'var(--bg-lighter)', borderRadius: '12px' }}>
                <Loader2 className="animate-spin" size={48} style={{ margin: '0 auto', marginBottom: '20px', color: 'var(--primary)' }} />
                <h3 style={{ marginBottom: '8px', color: '#fff' }}>
                  {libraryError ? "Connecting to AI Engine..." : "Loading Library..."}
                </h3>
                <p style={{ fontSize: '14px', maxWidth: '400px', margin: '0 auto' }}>
                  {libraryError 
                    ? "The massive 4.8 GB neural models are currently loading into your MacBook's GPU. This usually takes about 60 seconds on the first boot." 
                    : "Fetching cinematic stills..."}
                </p>
              </div>
            ) : (
              <div className="gallery-grid">
                {library.map((item, idx) => (
                  <div 
                    key={idx} 
                    className={`gallery-item ${selectedRef === item ? 'selected' : ''}`}
                    onClick={() => setSelectedRef(item)}
                  >
                    <img src={`${API_BASE}${item.path}`} alt={item.name} />
                    <div className="palette-overlay">
                      <span style={{ fontSize: '12px', fontWeight: '600' }}>{item.name}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          <motion.div className="glass-panel" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <h3>2. Upload Target Media</h3>
            <p className="text-muted" style={{ marginBottom: '16px', fontSize: '0.9rem' }}>
              Upload your target video (.mp4, .mov) or image.
            </p>

            <div 
              className="dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                accept="video/*,image/*,.dng,.cr2,.arw,.nef"
                onChange={handleFileSelect}
              />
              <div className="dropzone-content">
                {targetFile ? (
                  <>
                    <CheckCircle className="dropzone-icon" style={{ color: 'var(--accent-primary)' }} />
                    <span style={{ color: 'var(--text-main)', fontWeight: '600' }}>{targetFile.name}</span>
                  </>
                ) : (
                  <>
                    <UploadCloud className="dropzone-icon" />
                    <span>Drag & Drop your footage here or click to browse</span>
                  </>
                )}
              </div>
            </div>

            {error && (
              <div style={{ padding: '12px', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', marginTop: '16px' }}>
                {error}
              </div>
            )}

            <button 
              className="btn-primary" 
              onClick={handleGrade} 
              disabled={!selectedRef || !targetFile || isProcessing}
            >
              {isProcessing ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <Loader2 className="spinner" style={{ width: '20px', height: '20px', border: 'none', animation: 'spin 2s linear infinite' }} /> 
                  Rendering Masterpiece...
                </span>
              ) : (
                '🎨 Grade Footage'
              )}
            </button>
          </motion.div>
        </div>
      )}

      {result && (
        <motion.div className="glass-panel" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <h2>Masterpiece Rendered</h2>
            <button className="btn-primary" style={{ width: 'auto', marginTop: 0 }} onClick={() => setResult(null)}>Grade Another</button>
          </div>
          
          <div style={{ width: '100%', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)', background: '#000' }}>
            {result.type === 'video' ? (
              <video src={`${API_BASE}/api/download?path=${encodeURIComponent(result.output_media)}`} controls autoPlay loop style={{ width: '100%', display: 'block' }} />
            ) : (
              <img src={`${API_BASE}/api/download?path=${encodeURIComponent(result.output_media)}`} alt="Graded result" style={{ width: '100%', display: 'block' }} />
            )}
          </div>

          <div style={{ marginTop: '24px', display: 'flex', gap: '16px' }}>
            <a href={`${API_BASE}/api/download?path=${encodeURIComponent(result.output_media)}`} download className="btn-primary" style={{ textDecoration: 'none', display: 'flex', justifyContent: 'center', gap: '8px' }}>
              <Download /> Download Media
            </a>
            <a href={`${API_BASE}/api/download?path=${encodeURIComponent(result.output_lut)}`} download className="btn-primary" style={{ textDecoration: 'none', background: 'var(--bg-card-hover)', display: 'flex', justifyContent: 'center', gap: '8px' }}>
              <Film /> Download 3D LUT (.cube)
            </a>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default App;
