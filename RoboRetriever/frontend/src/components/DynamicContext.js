"use client"

import React, { createContext, useState,useContext ,useEffect} from 'react';
import UrlContext from '../UrlContext';



const DynamicContext = createContext();

export const DynamicProvider = ({ children}) => {
    const url = useContext(UrlContext)
    

    useEffect(() => {
        console.log("URL prop: ", url);
      }, [url]);

    const [isDynamic, setIsDynamic] = useState(false);


    const handleSwitchChange = async () => {
        const updatedIsDynamic = !isDynamic;
        setIsDynamic(updatedIsDynamic);

        try {
        const response = await fetch(url + "/change_mode", {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({ isDynamic: updatedIsDynamic }),
        });

        if (!response.ok) {
            throw new Error('Failed to update dynamic state');
        }

        console.log('Dynamic state updated successfully');
        } catch (error) {
        console.error('Error updating dynamic state:', error);
        }
    };

    return (
        <DynamicContext.Provider value={{ isDynamic, handleSwitchChange }}>
        {children}
        </DynamicContext.Provider>
    );
};

export default DynamicContext;