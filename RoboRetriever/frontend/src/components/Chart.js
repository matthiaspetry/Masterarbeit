"use client"
import React, { useState, useEffect } from 'react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart ,Bar,Rectangle,ResponsiveContainer,Cell} from 'recharts';

const Chart = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [lineChartData, setLineChartData] = useState([]);
  const [scatterChartData, setScatterChartData] = useState([]);
  const [lastLineDataPoint, setLastLineDataPoint] = useState(null);
  const [lastScatterDataPoint, setLastScatterDataPoint] = useState(null);

  const data01 = [
    { x: 0, y: 0 },
  ];

  const data = [
    {
      name: 'Cross',
      Count: 20,
    },
    {
      name: 'Cube',
      Count: 31,
    },
    {
      name: 'Hexagon',
      Count: 12,
   
    },
    {
      name: 'Cylinder',
      Count: 5,
      
    },
    {
      name: 'Pyramid',
      Count: 18,
      
    },
    {
      name: 'Star',
      Count: 3,
     
    },
    {
      name: 'Y_Cube',
      Count: 23,
     
    },
    
  ];

  const colors = ['#ff7f50', '#ff2717', '#0069e3', '#6c5ce7', '#82ca9d', '#ffff00',"#ffc658"];

const getColor = (index) => {
  return colors[index % colors.length];
};
  


  const handleTabClick = (index) => {
    setActiveTab(index);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const lineDataResponse = await fetchLineChartData(); // Fetch line chart data from backend
        const scatterDataResponse = await fetchScatterChartData(); // Fetch scatter plot data from backend
        console.log(lineDataResponse)
        console.log(scatterDataResponse)
       
  
        // Update line chart data
        // Update line chart data (check for empty response)
        if (lineDataResponse && !isDataEqual(lastLineDataPoint, lineDataResponse)) {
          setLineChartData(prevData => [...prevData, lineDataResponse]);
          setLastLineDataPoint(lineDataResponse);
        }

  
        // Update scatter plot data (check for empty response)
        if (scatterDataResponse && !isDataEqual(lastScatterDataPoint, scatterDataResponse)) {
          setScatterChartData(prevData => [...prevData, scatterDataResponse]);
          setLastScatterDataPoint(scatterDataResponse);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    const fetchLineChartData = async () => {
      const response = await fetch('http://127.0.0.1:5000/dataTime'); // Fetch line chart data from backend
      return await response.json();
    };

    const fetchScatterChartData = async () => {
      const response = await fetch('http://127.0.0.1:5000/dataPosition'); // Fetch scatter plot data from backend
      return await response.json();
    };

    const isDataEqual = (data1, data2) => {
      // Check if two data points are equal
      return JSON.stringify(data1) === JSON.stringify(data2);
    };

    const intervalId = setInterval(fetchData, 5000); // Fetch data every second
    return () => clearInterval(intervalId); // Cleanup interval on component unmount
  }, [lineChartData, scatterChartData]);

  return (
    <div className="flex">
      <div className="flex flex-col p-4">
        <div className="text-lg font-semibold mb-4">Charts</div>
        <button className={activeTab === 0 ? "tab tab-active" : "tab"} onClick={() => handleTabClick(0)}>Pick-up Time</button>
        <button className={activeTab === 1 ? "tab tab-active" : "tab"} onClick={() => handleTabClick(1)}>Pick-up Location</button>
        <button className={activeTab === 2 ? "tab tab-active" : "tab"} onClick={() => handleTabClick(2)}>Object Count</button>

      </div>
      <div className="tab-content p-4">
        {activeTab === 0 && (
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Pick-up Time</h3>
            
            {lineChartData.length > 0 && (  // Check if lineChartData has data
            <LineChart width={1000} height={580}  data={lineChartData}>
              <Line type="monotone" dataKey="Time" stroke="#8884d8" />
              <CartesianGrid stroke="#ccc" />
              <XAxis 
                label={{ value:"Objects", offset:-5, position:"insideBottom" }}
                dataKey="name" />
              <YAxis 
                label={{
                  value: `Time in Seconds`,
                  style: { textAnchor: 'middle' },
                  angle: -90,
                  position: 'left',
                  offset: 0,
                }}/>
              <Tooltip />
            </LineChart>
          )}
          {lineChartData.length === 0 && (  // Display message if no data
            <p>No pick-up time data available.</p>
          )}
          </div>
         
        )}
        {activeTab === 1 && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Pick-up Location</h3>
            {scatterChartData.length > 0 && (  // Check if scatterChartData has data
              <ScatterChart width={1000} height={580}  >
                <XAxis type="number" dataKey="x" name="x" domain={[-0.5, 0.5]}/>
                <YAxis dataKey="y" name="y" domain={[-0.5, 0]} />
                <CartesianGrid />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Pick-up Location" data={scatterChartData} fill="#8884d8" />
                <Scatter name="Robot Arm" data={data01} fill="#8884d8" shape="star" />
                <Legend />
              </ScatterChart>
            )}
            {scatterChartData.length === 0 && (  // Display message if no data
              <p>No pick-up location data available.</p>
            )}
          </div>
        )}
        {activeTab === 2 && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Object Count</h3>
            <BarChart
            width={1000} height={580} 
          data={data}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar
            dataKey="Count"    >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getColor(index)}
                stroke={getColor(index)}
              />
            ))}
          </Bar>
          
        </BarChart>
          </div>
        )}
      </div>
    </div>
  );
};

export default Chart;
