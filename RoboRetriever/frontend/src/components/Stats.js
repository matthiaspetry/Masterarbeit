import React from 'react'

const Stats = () => {
  return (
    <div className='h-fit'>
    <h3 className="text-xl font-semibold mb-4">Stats</h3>
    
    <div className="grid grid-cols-2 gap-2">
      <span className="font-medium">Success Rate:</span>
      <span className="font-extrabold text-right">
        94,25%
      </span>

    
      
      <span className="font-medium">Average Grasping Time:</span>
      <span className="font-extrabold text-right">
       3.54s
      </span>

      <span className="font-medium">Number of Objects:</span>
      <span className="font-extrabold text-right">
       23
      </span>



    </div>
  </div>
  )
}

export default Stats