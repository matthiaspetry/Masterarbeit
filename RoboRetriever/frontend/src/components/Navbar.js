"use client"
import React, { useState,useContext } from 'react'
import { Switch } from '@tremor/react';
import DynamicContext from './DynamicContext'; // Import the DynamicContext






function Navbar() {
    const { isDynamic, handleSwitchChange } = useContext(DynamicContext); // Use the useContext hook 


  return (
   
<nav className="bg-gray-200">
    <div className="grid grid-cols-5 grid-rows-1 gap-4  ">
        <div className="col-span-1 col-start-3 h-full">
    <div className="mx-auto px-4 flex">
        <div className="flex items-center py-5 px-2 text-gray-700 hover:text-gray-900">
            <svg  className="h-8 w-8 mr-2"  viewBox="0 0 64.989 61.649" ><path d="M14.789 8.902l-9.245 2.904 1.934 6.157c.882.319 1.812.484 2.75.489 2.567 0 4.987-1.199 6.543-3.242l-1.981-6.309zm2.615 8.325c-1.886 1.935-4.474 3.026-7.176 3.027-.71-.008-1.417-.092-2.109-.25l7.96 25.344c2.104-1.767 4.859-2.839 7.874-2.839a12.58 12.58 0 0 1 1.417.08l-7.966-25.362z"/><path d="M8.935 5.122L7.12 15.278l8.138 1.455c2.014-1.557 3.193-3.96 3.194-6.506-.003-1.229-.282-2.442-.816-3.55L8.935 5.122zm10.792 1.929c.346 1.023.524 2.096.528 3.176 0 2.57-.988 5.042-2.758 6.905l26.338 4.708 1.816-10.156-25.924-4.634z"/><circle cx="10.227" cy="10.227" r="8.086"/><path d="M23.954 44.484c-5.247 0-9.501 4.083-9.501 9.119-.093 2.115-.094 4.311-.108 6.134h19.218c.011-1.987.038-4.325-.108-6.134 0-5.036-4.254-9.119-9.501-9.119zm18.992-23.018l8.22 7.728 11.033-2.36-.829-3.876-9.002 1.926-6.707-6.306zm1.855-10.298l10.378-4.426 9.546 6.013-2.112 3.354-7.789-4.907-8.467 3.612z"/><path d="M41.376 9.166l7.17 1.134-2.132 13.482-7.171-1.134z"/></svg>
            <span className="text-xl font-bold">VisionGraspNet Interface</span>
        </div>
        
    </div>
    </div>
    <div className="col-span-1 col-start-5 h-full flex items-center"> {/* Added flex items-center */}
          <div className="mx-auto px-4 flex">
            <Switch
              checked={isDynamic}
              onChange={handleSwitchChange}
            />
<span className="ml-2 font-bold text-lg">{isDynamic ? 'Dynamic' : 'Static'}</span>
          </div>
        </div>
    </div>

</nav>
  )
}

export default Navbar