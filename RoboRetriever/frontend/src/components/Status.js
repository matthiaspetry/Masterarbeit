"use client"
import React, { useState, useEffect , useContext } from 'react';
import { Badge,BadgeDelta } from "@tremor/react";
import UrlContext from '../UrlContext';
import DynamicContext from './DynamicContext'; // Import the DynamicContext



const Status = () => {
  const url = useContext(UrlContext);
  const { isDynamic } = useContext(DynamicContext);

  const [statusData, setStatusData] = useState({
    isConnected: false,
    operationalStatus: false, // Updated to use boolean
    currentTask: "",
    errorStatus: ""
  });
  useEffect(() => {
    console.log("URL prop: ", url);
    const apiUrl = url + '/status';
    console.log(apiUrl)
    const fetchData = async () => {
      try {
        const response = await fetch(apiUrl);
        const data = await response.json();
        console.log( data.objectPlacePosition)
        setStatusData({
          isConnected: data.isConnected,
          gripperConnection: data.gripperConnection,
          operationalStatus: data.operationalStatus,
          currentTask: data.currentTask,
          errorStatus: data.errorStatus,
          objectPickedUp: data.objectPickedUp,
          selectedSpeed: data.selectedSpeed,
          selectedModel: data.selectedModel
        });
      } catch (error) {
        console.error('Error fetching data: ', error);
      }
    };

    // Call fetchData every 1000 milliseconds (1 second)
    const intervalId = setInterval(fetchData, 1000);
  

    // Cleanup interval on component unmount
    return () => clearInterval(intervalId);
  }, [url,isDynamic]); // Empty dependency array ensures this effect runs only once on mount

  return (
    <div>
      <h3 className="text-xl font-semibold mb-4">Robotic Arm Status</h3>
      
      <div className="grid grid-cols-2 gap-2">
        <span className="font-medium">Robot Arm:</span>
        <span className="text-right">
          {statusData.isConnected ? <Badge color="green" size="xl">Connected</Badge> : <Badge color="red" size="xl">Disconnected</Badge>}
        </span>

      
        
        <span className="font-medium">Gripper:</span>
        <span className="text-right">
        {statusData.gripperConnection ? <Badge color="green" size="xl">Connected</Badge> : <Badge color="red" size="xl">Disconnected</Badge>}
        </span>

        <span className="font-medium">Operational Status:</span>
        <span className="text-right">
            {statusData.operationalStatus === 'FreeDrive' ? 
                <Badge color="blue" size="xl">Free Drive</Badge> :
                statusData.operationalStatus ? 
                    <Badge color="green" size="xl">Active</Badge> : 
                    <Badge color="red" size="xl">Inactive</Badge>
            }
        </span>

        <span className="font-medium">Selected Object:</span>
        <span className="text-right">
            <Badge size="xl"> {statusData.currentTask}</Badge>
        </span>
        
        <span className="font-medium">Object Picked:</span>
        <span className="text-right">
          {statusData.objectPickedUp ? <Badge  size="xl">Yes</Badge> : <Badge size="xl">No</Badge>}

        </span>

        <span className="font-medium">Conveyor Speed:</span>
        <span className="text-right">
        <Badge size="xl"> {statusData.selectedSpeed}</Badge>

        </span>
        <span className="font-medium">Model:</span>
        <span className="text-right">
        <Badge size="xl"> {statusData.selectedModel}</Badge>

        </span>

  
      </div>
    </div>
  );

};

export default Status;
