"use client"
import Image from 'next/image'
import DynamicContext from './DynamicContext'; // Import the DynamicContext
import React, {useContext ,useEffect} from 'react';


const ExampleGif = () => {
  const { isDynamic } = useContext(DynamicContext);

  return (
    <div>
       {isDynamic && (
                        <Image src="/ur3e.gif" width={700} height={700}alt="Example GIF" />
                    )}
     {!isDynamic && (
                        <Image src="/static.jpg" width={700} height={700}alt="Example GIF" />
                    )}

    </div>


  );
};

export default ExampleGif;