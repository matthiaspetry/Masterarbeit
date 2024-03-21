"use client"
import { useState } from 'react';
import Image from 'next/image';
import { Select, SelectItem } from '@tremor/react';

const Control = () => {
    const apiUrl = 'http://localhost:5000';  // Replace with your Flask server URL
    const [selectedImage, setSelectedImage] = useState('');

    // Assuming you store your images in the public folder in Next.js
    const objectImages = {
        "0": "/Users/matthiaspetry/Desktop/Masterarbeit/RoboRetriever/frontend/public/cross.jpeg", // Adjust paths based on your actual images location
        "1": "/Users/matthiaspetry/Desktop/Masterarbeit/RoboRetriever/frontend/public/redcube.jpg",
        "2": "/images/cylinder.png",
        "3": "/images/hexagon.png",
        "4": "/Users/matthiaspetry/Desktop/Masterarbeit/RoboRetriever/frontend/public/Pyramid.jpg",
        "5": "/images/r_cylinder.png",
        "6": "/images/star.png",
        "7": "/images/y_cube.png",
    };

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

    const free_mode = () => {
        fetch(`${apiUrl}/free_mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const set_position = () => {
        fetch(`${apiUrl}/set_position`, {
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
    const handleDropdownObject = (event) => {
        setObject(event)
        const selectedValue = event
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

    const handleDropdownModel = (event) => {
        setModel(event)
        const selectedValue = event;
        console.log("Selected Value:", selectedValue);

        setSelectedImage(objectImages[selectedValue] || '');
    
        fetch(`${apiUrl}/select_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ object_type: selectedValue })  // Send the selected value in the request body
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const handleDropdownSpeed = (event) => {
        setSpeed(event)
        const selectedValue = event;
        console.log("Selected Value:", selectedValue);

    
        fetch(`${apiUrl}/select_speed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ object_type: selectedValue })  // Send the selected value in the request body
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    };

    const [object, setObject] = useState('');
    const [model, setModel] = useState('');
    const [speed, setSpeed] = useState('');


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
                <button onClick={free_mode} className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded">
                    Free Drive
                </button>
                <button onClick={set_position} className="bg-indigo-500 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded">
                    Set Position
                </button>

            

                <Select id="object" name="object" placeholder="Select an Object" value={object} onValueChange={handleDropdownObject} className="text-xl font-bold">
                    <SelectItem value="0" >
                    Cross
                    </SelectItem>
                    <SelectItem value="1" >
                    Cube
                    </SelectItem>
                    <SelectItem value="2" >
                    Cylinder
                    </SelectItem>
                    <SelectItem value="3">
                    Hexagon
                    </SelectItem>
                    <SelectItem value="4">
                    Pyramid
                    </SelectItem>
                    <SelectItem value="5">
                    Star
                    </SelectItem>
                    <SelectItem value="6">
                    Y_Cube
                    </SelectItem>
                </Select>

                <Select id="model" name="model" placeholder="Select a Model" value={model} onValueChange={handleDropdownModel} className="text-xl font-semibold">
                    <SelectItem value="FastVit Backbone" >
                    FastVit Backbone
                    </SelectItem>
                    <SelectItem value="ResNet18 Backbone" >
                    ResNet18 Backbone
                    </SelectItem>
                    <SelectItem value="CNN Backbone" >
                    CNN Backbone
                    </SelectItem>
                </Select>

                <Select id="speed" name="speed" placeholder="Set Conveyor Speed" value={speed} onValueChange={handleDropdownSpeed} className=" text-xl font-semibold">
                    <SelectItem value="66 mm/s" >
                    66 mm/s
                    </SelectItem>
                    <SelectItem value="120 mm/s" >
                    120 mm/s
                    </SelectItem>
                    <SelectItem value="150 mm/s" >
                    150 mm/s
                    </SelectItem>
                    <SelectItem value="200 mm/s" >
                    200 mm/s
                    </SelectItem>
                </Select>


               
               
            </div>
            
            
        </div>
      </div>
        
    );
};

export default Control
