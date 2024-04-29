"use client"
import React, { useState, useEffect , useContext } from 'react';
import UrlContext from '../UrlContext';

const Stats = () => {
  const [statsData, setStatsData] = useState(null);
  const url = useContext(UrlContext);
  

  useEffect(() => {
    const apiUrl = url + '/stats';
    const fetchData = () => {
      fetch(apiUrl) // assuming your Flask server is running on the same domain
        .then(response => response.json())
        .then(data => setStatsData(data))
        .catch(error => console.error('Error fetching stats:', error));
    };

    // Fetch data initially
    fetchData();

    // Fetch data every 1 second
    const interval = setInterval(fetchData, 1000);

    // Clean up interval on unmount
    return () => clearInterval(interval);
  }, [url]);

  return (
    <div className='h-fit'>
      <h3 className="text-xl font-semibold mb-4">Stats</h3>
      {statsData && (
        <div className="grid grid-cols-2 gap-2">
          <span className="font-medium">Success Rate Complete:</span>
          <span className="font-extrabold text-right">
            {statsData.totalSuccessRate.toFixed(2)}%
          </span>
          <span className="font-medium">Success Rate Static:</span>
          <span className="font-extrabold text-right">
            {statsData.successRateStatic.toFixed(2)}%
          </span>
          <span className="font-medium">Success Rate Dynamic:</span>
          <span className="font-extrabold text-right">
            {statsData.successRateDynamic.toFixed(2)}%
          </span>
          <span className="font-medium">Average Grasping Time:</span>
          <span className="font-extrabold text-right">
            {statsData.avgGraspTime.toFixed(2)}s
          </span>
          <span className="font-medium">Number of Objects:</span>
          <span className="font-extrabold text-right">
            {statsData.totalCount}
          </span>
        </div>
      )}
    </div>
  );
};

export default Stats;
