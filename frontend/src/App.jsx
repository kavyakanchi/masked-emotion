import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // Camera state
  const [isCameraMode, setIsCameraMode] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreviewUrl(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
      setIsCameraMode(false)
    }
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      setIsCameraMode(true)
      setFile(null)
      setPreviewUrl(null)
      setResult(null)
      setError(null)
    } catch (err) {
      setError("Failed to access camera. Please allow camera permissions.")
    }
  }

  // When camera mode turns on, attach the stream to the video element after it renders
  useEffect(() => {
    if (isCameraMode && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [isCameraMode])

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setIsCameraMode(false)
  }

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas')
      // Ensure we have valid dimensions
      if (videoRef.current.videoWidth === 0) {
        setError("Camera not ready yet. Please try again in a second.")
        return
      }
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      
      // The video is mirrored via CSS, but we'll capture it normally.
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      
      canvas.toBlob((blob) => {
        if (!blob) {
            setError("Failed to capture image.")
            return
        }
        const capturedFile = new File([blob], "capture.jpg", { type: "image/jpeg" })
        setFile(capturedFile)
        setPreviewUrl(URL.createObjectURL(capturedFile))
        stopCamera()
      }, 'image/jpeg')
    }
  }

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Use environment variable for production, fallback to localhost for development
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to predict emotion')
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="bg-gradients">
        <div className="blob-1"></div>
        <div className="blob-2"></div>
      </div>

      <div className="app-container">
        <header className="header">
          <h1>MaskSense AI</h1>
          <p>Real-time Emotion Recognition for Masked Faces</p>
        </header>

        <main className="main-content">
          <section className="upload-section">
            
            <div className="upload-box">
              {isCameraMode ? (
                <>
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted
                    className="camera-video"
                  ></video>
                </>
              ) : previewUrl ? (
                <img src={previewUrl} alt="Preview" className="preview-image" />
              ) : (
                <>
                  <input 
                    type="file" 
                    className="file-input" 
                    accept="image/*" 
                    onChange={handleFileChange}
                  />
                  <div className="upload-icon">📸</div>
                  <p>Click or drag an image here</p>
                  <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Supports JPG, PNG</p>
                </>
              )}
            </div>

            <div className="action-buttons">
              {isCameraMode ? (
                <>
                  <button className="btn-secondary" onClick={capturePhoto} style={{flex: 1}}>
                    Snap Photo
                  </button>
                  <button className="btn-secondary" onClick={stopCamera} style={{flex: 1, backgroundColor: 'rgba(255, 68, 68, 0.1)', borderColor: 'rgba(255, 68, 68, 0.3)', color: '#ff4444'}}>
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button className="btn-secondary" onClick={startCamera}>
                    Live Camera
                  </button>
                  {file && (
                    <button 
                      className="btn-secondary" 
                      onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); setError(null); }}
                      style={{backgroundColor: 'rgba(255, 68, 68, 0.1)', borderColor: 'rgba(255, 68, 68, 0.3)', color: '#ff4444'}}
                    >
                      Clear
                    </button>
                  )}
                </>
              )}

              <button 
                className="btn-primary" 
                onClick={handlePredict} 
                disabled={!file || loading || isCameraMode}
              >
                {loading ? 'Analyzing...' : 'Predict Emotion'}
              </button>
            </div>

            {error && (
              <p style={{ color: '#ff4444', marginTop: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>
                {error}
              </p>
            )}
          </section>

          <section className="result-section">
            {loading ? (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Running CNN Model...</p>
              </div>
            ) : result ? (
              <div className="emotion-result">
                <h2>Predicted Emotion</h2>
                <div className="emotion-name">{result.emotion}</div>
                
                <div style={{ width: '100%', maxWidth: '300px', marginTop: '2rem', margin: '2rem auto 0 auto' }}>
                  <div className="confidence-bar-bg">
                    <div 
                      className="confidence-bar-fill" 
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                  <div className="confidence-text">
                    Confidence: {result.confidence}%
                  </div>
                </div>
              </div>
            ) : (
              <div className="result-placeholder">
                <p>Upload an image or take a photo to see the results here.</p>
              </div>
            )}
          </section>
        </main>
      </div>
    </>
  )
}

export default App
