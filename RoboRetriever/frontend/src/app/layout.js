"use client"
import React from 'react';

import { useState } from 'react';
import UrlContext from '../UrlContext.js';
import { Inter } from "next/font/google";
import "./globals.css";
import { DynamicProvider } from '@/components/DynamicContext'; // Import the DynamicProvider


import Navbar from "@/components/Navbar";
import Overlay from "@/components/Overlay";
const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({ children }) {
  const [url, setUrl] = useState('');
  const [isOverlayVisible, setIsOverlayVisible] = useState(true);

  const handleUrlSubmit = (url) => {
    // Do something with the URL
    console.log('URL submitted:', url);
    setUrl(url); // Update the URL state
    setIsOverlayVisible(false); // Hide the overlay
  };

  return (
      <html lang="en">
        <head>
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </head>
        <body className="h-full">
        <UrlContext.Provider value={url}>
          <DynamicProvider value={url}>
          <Navbar />
          {isOverlayVisible ? <Overlay onUrlSubmit={handleUrlSubmit} /> : null}
          
              {children}
          
          </DynamicProvider>
          </UrlContext.Provider>
        </body>
      </html>
  );
}
