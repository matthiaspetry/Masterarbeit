"use client"
import React, { useContext, useEffect } from 'react';
import UrlContext from '../UrlContext';



const Livefeed = () => {
  const url = useContext(UrlContext);

  useEffect(() => {
    console.log("URL prop: ", url);
  }, [url]);


  return (
    <div>
      <h2 className=" text-lg font-semibold mb-4 ">Live Feed</h2>
      <div className="relative" style={{ maxWidth: '800px' }}>
        {/* Adjust the maxWidth to your desired width */}
        <img
          src={url+"/video_feed"}
          alt="Live Feed"
          style={{ width: '100%', height: '100' }}
        />
      </div>
    </div>
  );
};

export default Livefeed;