"use client"
const Control = () => {
    const apiUrl = 'http://localhost:5000';  // Replace with your Flask server URL

    const start_processing = () => {
        fetch(`${apiUrl}/start_processing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const connect_robot = () => {
        fetch(`${apiUrl}/connect_robot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const stop_processing = () => {
        fetch(`${apiUrl}/stop_processing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const move_2_base = () => {
        fetch(`${apiUrl}/move_2_base`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    // Function to handle dropdown change
    const handleDropdownChange = (event) => {
        const selectedValue = event.target.value;
        console.log("Selected Value:", selectedValue);
    
        fetch(`${apiUrl}/select_object`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ object_type: selectedValue })  // Send the selected value in the request body
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    return (
        <div>
        <h2 className="text-lg font-semibold mb-4">Robot Control</h2>
        <div >
        <div className="grid grid-cols-2 gap-4">
                <button onClick={start_processing} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Start Processing
                </button>
                <button onClick={connect_robot} className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                    Connect Robot
                </button>
                <button onClick={stop_processing} className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded">
                    Stop Processing
                </button>
                <button onClick={move_2_base} className="bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded">
                    Back to Base
                </button>

            </div>
            <div className="grid grid-cols-1 gap-4 py-4">
                <select onChange={handleDropdownChange} className="bg-gray-200 text-gray-700 font-bold py-2 px-4 rounded w-full">
                    <option value="">Select an Option</option>
                    <option value="0">Cross</option>
                    <option value="1">Cube</option>
                    <option value="2">Cylinder</option>
                    <option value="3">Hexagon</option>
                    <option value="4">Pyramid</option>
                    <option value="5">R_Cylinder</option>
                    <option value="6">Star</option>
                    <option value="7">Y_Cube</option>
                </select>
            </div>
            
        </div>
      </div>
        
    );
};

export default Control
