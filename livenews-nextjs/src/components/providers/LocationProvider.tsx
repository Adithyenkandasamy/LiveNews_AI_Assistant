'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';

interface LocationData {
  country: string;
  city: string;
  coords?: {
    latitude: number;
    longitude: number;
  };
}

interface LocationContextType {
  location: LocationData | null;
  requestLocation: () => Promise<void>;
  setManualLocation: (location: LocationData) => void;
  isLoading: boolean;
}

const LocationContext = createContext<LocationContextType | undefined>(undefined);

export function LocationProvider({ children }: { children: React.ReactNode }) {
  const [location, setLocation] = useState<LocationData | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Try to get location from localStorage first
    const savedLocation = localStorage.getItem('livenews-location');
    if (savedLocation) {
      setLocation(JSON.parse(savedLocation));
    } else {
      // Auto-request location on first visit
      requestLocation();
    }
  }, []);

  const requestLocation = async () => {
    setIsLoading(true);
    
    try {
      // First try browser geolocation
      if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
          async (position) => {
            const { latitude, longitude } = position.coords;
            
            // Reverse geocode to get location details
            try {
              const response = await fetch(`/api/location/reverse?lat=${latitude}&lng=${longitude}`);
              if (response.ok) {
                const locationData = await response.json();
                const newLocation = {
                  country: locationData.country,
                  city: locationData.city,
                  coords: { latitude, longitude }
                };
                setLocation(newLocation);
                localStorage.setItem('livenews-location', JSON.stringify(newLocation));
              }
            } catch (error) {
              console.error('Reverse geocoding failed:', error);
              // Fallback to IP-based location
              await getLocationFromIP();
            }
            setIsLoading(false);
          },
          async (error) => {
            console.warn('Geolocation denied:', error);
            // Fallback to IP-based location
            await getLocationFromIP();
            setIsLoading(false);
          }
        );
      } else {
        // Fallback to IP-based location
        await getLocationFromIP();
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Location request failed:', error);
      setIsLoading(false);
    }
  };

  const getLocationFromIP = async () => {
    try {
      const response = await fetch('/api/location/ip');
      if (response.ok) {
        const locationData = await response.json();
        const newLocation = {
          country: locationData.country,
          city: locationData.city,
        };
        setLocation(newLocation);
        localStorage.setItem('livenews-location', JSON.stringify(newLocation));
      }
    } catch (error) {
      console.error('IP location failed:', error);
      // Set default location
      const defaultLocation = { country: 'US', city: 'New York' };
      setLocation(defaultLocation);
      localStorage.setItem('livenews-location', JSON.stringify(defaultLocation));
    }
  };

  const setManualLocation = (newLocation: LocationData) => {
    setLocation(newLocation);
    localStorage.setItem('livenews-location', JSON.stringify(newLocation));
  };

  return (
    <LocationContext.Provider value={{
      location,
      requestLocation,
      setManualLocation,
      isLoading
    }}>
      {children}
    </LocationContext.Provider>
  );
}

export function useLocation() {
  const context = useContext(LocationContext);
  if (context === undefined) {
    throw new Error('useLocation must be used within a LocationProvider');
  }
  return context;
}
