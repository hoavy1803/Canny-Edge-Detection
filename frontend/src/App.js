import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [edgeImage, setEdgeImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedKernel, setSelectedKernel] = useState('');
  const [selectedLowThreshold, setSelectedLowThreshold] = useState('');
  const [selectedHighThreshold, setSelectedHighThreshold] = useState('');


  const handleFileChange = (event) => {
    const file = event.target.files[0]
    setSelectedFile(event.target.files[0]);
    const reader = new FileReader();
    reader.onload = () => {
      setOriginalImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('selectedKernel', selectedKernel);
    formData.append('selectedLowThreshold', selectedLowThreshold);
    formData.append('selectedHighThreshold', selectedHighThreshold);
    setLoading(true); // Bắt đầu hiển thị loading
    axios.post('http://localhost:8000/edge-detection', formData, { responseType: 'blob' })
      .then(response => {
        const imageBlob = new Blob([response.data], { type: 'image/png' });
        const imageUrl = URL.createObjectURL(imageBlob);
        setEdgeImage(imageUrl);
        setLoading(false); // Kết thúc hiển thị loading
      })
      .catch(error => {
        console.error(error);
        setLoading(false); // Kết thúc hiển thị loading
      });
  };

  // Hàm xử lý khi người dùng nhấp vào nút tải xuống
  const handleDownload = () => {
    if (edgeImage) {
      // Tạo một liên kết tạm thời để tải xuống ảnh
      const link = document.createElement('a');
      link.href = edgeImage;
      link.download = 'edge_image.jpg';
      link.click();
    }
    alert("No edge image to download!")
  };

  const [coordinates1, setCoordinates1] = useState({ x: 0, y: 0 });
  const [coordinates2, setCoordinates2] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event, imageNumber) => {
    const { clientX, clientY } = event;
    const { left, top } = event.target.getBoundingClientRect();
    const x = parseInt(clientX - left);
    const y = parseInt(clientY - top);

    if (imageNumber === 1) {
      setCoordinates1({ x, y });
    } else if (imageNumber === 2) {
      setCoordinates2({ x, y });
    };
  };

  const [grayValue1, setGrayValue1] = useState(0);
  const [grayValue2, setGrayValue2] = useState(0);

  const getGrayValue = (e, imageNumber) => {
    const image = e.target;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0, image.width, image.height);

    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;

    const pixelData = context.getImageData(x, y, 1, 1).data;
    const gray = (pixelData[0] + pixelData[1] + pixelData[2]) / 3;


    if (imageNumber === 1) {
      setGrayValue1(gray);
    } else if (imageNumber === 2) {
      setGrayValue2(gray);
    };
  };

  const handleMouseMoves = (event, number) => {
    handleMouseMove(event, number);
    getGrayValue(event, number);
  };


  return (
    <div>
      <table className='interface'>
        <td id='menu-banner'>
          <tr>
            <tr><h3>Options:</h3></tr>

            <tr id='options'>
              <div className='option-item'>
                <td className='option-name'><p>Kernel size:</p></td>
                <td>
                  <input className='input'
                    type="number"
                    value={selectedKernel}
                    onChange={event => setSelectedKernel(event.target.value)}
                  />
                </td>
              </div>

            </tr>

            <tr>
              <div className='option-item'>
                <td className='option-name'><p>Low threshold:</p></td>
                <td>
                  <input className='input'
                    type="number"
                    value={selectedLowThreshold}
                    onChange={event => setSelectedLowThreshold(event.target.value)}
                  />
                </td>
              </div>

            </tr>

            <tr>
              <div className='option-item'>
                <td className='option-name'><p>High threshold:</p></td>
                <td>
                  <input className='input'
                    type="number"
                    value={selectedHighThreshold}
                    onChange={event => setSelectedHighThreshold(event.target.value)}
                  />
                </td>
              </div>
            </tr>
          </tr>

          <tr>
            <tr><input className='choosefile' type="file" onChange={handleFileChange} /></tr>
            <tr><button className='custom-button' onClick={handleUpload}>Upload and Process</button></tr>
            <tr><button className='custom-button' onClick={handleDownload}>Download Edge Image</button></tr>
          </tr>

          <tr><div id='contact'><a href='https://www.facebook.com/profile.php?id=100009777309118' target='blank'>Contact Us</a></div></tr>
        </td>

        <td id='main'>
          <tr className='banner'>

            <h1>Canny Edge Detector</h1>

          </tr>

          <tr className='main-container'>
            <td>
              {originalImage && (
                <div>

                  <div className='image-container'>
                    <h2>Original Image</h2>
                    <img src={originalImage} alt="Original Image" onMouseMove={(event) => handleMouseMoves(event, 1)} />
                    <p>Value at [X: {coordinates1.x}, Y: {coordinates1.y}] = {grayValue1}</p>
                  </div>
                </div>
              )}
            </td>

            <td>
              {loading ? (
                <div>
                  <div className='loading-spinner' />
                  <h3>Processing, please wait...</h3>
                </div>
              ) : (
                edgeImage && (
                  <div>
                    <div className='image-container'>
                      <h2>Edge Image</h2>
                      <img src={edgeImage} alt="Edge Image" onMouseMove={(event) => handleMouseMoves(event, 2)} />
                      <p>Value at [X: {coordinates2.x}, Y: {coordinates2.y}] = {grayValue2}</p>
                    </div>
                  </div>
                )
              )}
            </td>
          </tr>
        </td>
      </table>
    </div >
  );
}

export default App;