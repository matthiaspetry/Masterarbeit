"use client"
import React, { useState, useEffect } from 'react';

const Status = () => {
  const [statusData, setStatusData] = useState({
    isConnected: false,
    operationalStatus: false, // Updated to use boolean
    currentTask: "",
    errorStatus: ""
  });

  useEffect(() => {
    const apiUrl = 'http://127.0.0.1:5000/status';

    const fetchData = async () => {
      try {
        const response = await fetch(apiUrl);
        const data = await response.json();

        setStatusData({
          isConnected: data.isConnected,
          gripperConnection: data.gripperConnection,
          operationalStatus: data.operationalStatus,
          currentTask: data.currentTask,
          errorStatus: data.errorStatus,
          objectPickedUp: data.objectPickedUp
        });
      } catch (error) {
        console.error('Error fetching data: ', error);
      }
    };

    // Call fetchData every 1000 milliseconds (1 second)
    const intervalId = setInterval(fetchData, 1000);

    // Cleanup interval on component unmount
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures this effect runs only once on mount

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4">Robotic Arm Status</h3>
      
      <div className="mb-3">
        <span className="font-medium">Robot Arm:</span>
        <span className={`ml-2 ${statusData.isConnected ? 'text-green-600' : 'text-red-600'}`}>
          {statusData.isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      <div className="mb-3">
        <span className="font-medium">Gripper:</span>
        <span className={`ml-2 ${statusData.gripperConnection ? 'text-green-600' : 'text-red-600'}`}>
          {statusData.isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div className="mb-3">
        <span className="font-medium">Operational Status:</span>
        <span className={`ml-2 ${statusData.operationalStatus ? 'text-green-600' : 'text-red-600'}`}>
          {statusData.operationalStatus ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className="mb-3">
        <span className="font-medium">Selected Object:</span>
        <span className="ml-2">{statusData.currentTask}</span>
      </div>

      <div className="mb-3">
        <span className="font-medium">Error Status:</span>
        <span className="ml-2">{statusData.errorStatus}</span>
      </div>
      <div className="mb-3">
        <span className="font-medium">Object Picked:</span>
        <span className={`ml-2 ${statusData.objectPickedUp ? 'text-green-600' : 'text-red-600'}`}>
          {statusData.isConnected ? 'Yes' : 'No'}
        </span>
      </div>
    </div>
  );
};

export default Status;
