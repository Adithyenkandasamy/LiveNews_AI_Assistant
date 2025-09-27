'use client';

import { useState, useEffect } from 'react';
import { useLocation } from '@/components/providers/LocationProvider';
import { X, MapPin, Search, Navigation } from 'lucide-react';
import { motion } from 'framer-motion';

interface LocationSelectorProps {
  onClose: () => void;
}

const popularCities = [
  { city: 'New York', country: 'US', flag: '🇺🇸' },
  { city: 'London', country: 'GB', flag: '🇬🇧' },
  { city: 'Tokyo', country: 'JP', flag: '🇯🇵' },
  { city: 'Paris', country: 'FR', flag: '🇫🇷' },
  { city: 'Sydney', country: 'AU', flag: '🇦🇺' },
  { city: 'Mumbai', country: 'IN', flag: '🇮🇳' },
  { city: 'Toronto', country: 'CA', flag: '🇨🇦' },
  { city: 'Berlin', country: 'DE', flag: '🇩🇪' },
  { city: 'Singapore', country: 'SG', flag: '🇸🇬' },
  { city: 'Dubai', country: 'AE', flag: '🇦🇪' },
  { city: 'Beijing', country: 'CN', flag: '🇨🇳' },
  { city: 'São Paulo', country: 'BR', flag: '🇧🇷' },
];

export function LocationSelector({ onClose }: LocationSelectorProps) {
  const { location, setManualLocation, requestLocation, isLoading } = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleCitySelect = (city: string, country: string) => {
    setManualLocation({ city, country });
    onClose();
  };

  const handleLocationRequest = async () => {
    await requestLocation();
    onClose();
  };

  const searchLocations = async (query: string) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      // Mock search results for development
      const mockResults = popularCities.filter(
        city => 
          city.city.toLowerCase().includes(query.toLowerCase()) ||
          city.country.toLowerCase().includes(query.toLowerCase())
      );
      
      setSearchResults(mockResults);
    } catch (error) {
      console.error('Location search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      searchLocations(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  const filteredCities = popularCities.filter(city =>
    searchQuery === '' || 
    city.city.toLowerCase().includes(searchQuery.toLowerCase()) ||
    city.country.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold flex items-center gap-2">
                <MapPin size={24} />
                Select Location
              </h2>
              <p className="text-green-100 text-sm">
                Get news relevant to your location
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-white/80 hover:text-white hover:bg-white/20 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(80vh-120px)]">
          {/* Current Location */}
          {location && (
            <div className="mb-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
              <div className="flex items-center gap-3 mb-2">
                <MapPin className="text-blue-600" size={20} />
                <div>
                  <p className="font-medium text-blue-800">Current Location</p>
                  <p className="text-sm text-blue-600">{location.city}, {location.country}</p>
                </div>
              </div>
            </div>
          )}

          {/* Auto-detect Location */}
          <div className="mb-6">
            <button
              onClick={handleLocationRequest}
              disabled={isLoading}
              className="w-full flex items-center gap-3 p-4 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-xl hover:from-green-700 hover:to-blue-700 transition-all disabled:opacity-50"
            >
              <Navigation size={20} />
              <div className="text-left">
                <p className="font-medium">
                  {isLoading ? 'Detecting Location...' : 'Use My Current Location'}
                </p>
                <p className="text-sm text-green-100">
                  Automatically detect your location for local news
                </p>
              </div>
            </button>
          </div>

          {/* Search */}
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search for a city or country..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Search Results or Popular Cities */}
          <div>
            <h3 className="font-semibold text-gray-800 mb-3">
              {searchQuery ? 'Search Results' : 'Popular Locations'}
            </h3>
            
            {isSearching ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2">
                {(searchQuery ? searchResults : filteredCities).map((city, index) => (
                  <motion.button
                    key={`${city.city}-${city.country}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => handleCitySelect(city.city, city.country)}
                    className="flex items-center gap-3 p-3 text-left hover:bg-gray-50 rounded-lg transition-colors group"
                  >
                    <span className="text-2xl">{city.flag}</span>
                    <div className="flex-1">
                      <p className="font-medium text-gray-800 group-hover:text-blue-600">
                        {city.city}
                      </p>
                      <p className="text-sm text-gray-500">{city.country}</p>
                    </div>
                    <MapPin className="text-gray-400 group-hover:text-blue-600" size={16} />
                  </motion.button>
                ))}
              </div>
            )}

            {searchQuery && searchResults.length === 0 && !isSearching && (
              <div className="text-center py-8">
                <MapPin className="mx-auto text-gray-400 mb-2" size={48} />
                <p className="text-gray-500">No locations found</p>
                <p className="text-sm text-gray-400">Try searching for a different city or country</p>
              </div>
            )}
          </div>

          {/* Location Benefits */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-xs text-gray-500 mb-3 text-center">Why set your location?</p>
            <div className="space-y-2 text-xs text-gray-600">
              <div className="flex items-center gap-2">
                <span className="text-green-500">📰</span>
                <span>Get local news and regional coverage</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-500">🌡️</span>
                <span>Weather and local event updates</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-purple-500">📍</span>
                <span>Location-specific trending topics</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
