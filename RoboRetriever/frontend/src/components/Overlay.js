"use client"
import { useState } from 'react';

const Overlay = ({ onUrlSubmit }) => {
  const [url, setUrl] = useState('');
  const [isVisible, setIsVisible] = useState(true);

  const handleSubmit = (e) => {
    if (e.key === 'Enter') {
      onUrlSubmit(url);
      setIsVisible(false);
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="fixed top-0 left-0 h-full w-full z-10 flex items-center justify-center">
      {/* Added a new parent div with a semi-transparent background color using rgba() */}
      <div className="bg-gray-500 bg-opacity-50 h-full w-full flex items-center justify-center">
        <div className="bg-white p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4 text-center">Enter Bore URL</h2>
          <input
            type="text"
            className="border-2 border-gray-400 p-2 w-full rounded-md focus:outline-none focus:border-blue-500"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyPress={handleSubmit}
            placeholder="Enter URL"
          />
          <button
            className="bg-blue-500 text-white px-4 py-2 rounded-md mt-4 hover:bg-blue-600 focus:outline-none"
            onClick={() => { onUrlSubmit(url); setIsVisible(false); }}
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
};

export default Overlay
