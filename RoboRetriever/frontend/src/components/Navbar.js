import React from 'react'

function Navbar() {
  return (
    <nav className="bg-gray-200">
        <div className="mx-auto px-4">
            <div className="flex justify-between">
            <div className="flex space-x-4">
                <div>
                    <div className="flex items-center py-5 px-2 text-gray-700 hover:text-gray-900">
                        <svg className="h-6 w-6 mr-1 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                        <span className="font-bold">RoboRetriver Dashboard</span>
                    </div>
                </div>  
            </div>
            </div>
        </div>  
    </nav>
  )
}

export default Navbar