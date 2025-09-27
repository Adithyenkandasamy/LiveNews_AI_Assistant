'use client';

import { useState } from 'react';
import { useAuth } from '@/components/providers/AuthProvider';
import { useLocation } from '@/components/providers/LocationProvider';
import { User, MapPin, Bot, Settings, LogOut, Menu, X } from 'lucide-react';

interface HeaderProps {
  onShowAuth: () => void;
  onShowLocation: () => void;
  onShowAIBrief: () => void;
}

export function Header({ onShowAuth, onShowLocation, onShowAIBrief }: HeaderProps) {
  const { user, logout } = useAuth();
  const { location } = useLocation();
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-white/20 shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">வெ</span>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                வெளிச்சம்
              </h1>
              <p className="text-xs text-gray-500 hidden sm:block">Personalized Intelligence</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-4">
            {/* Location */}
            <button
              onClick={onShowLocation}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors text-sm"
            >
              <MapPin size={16} />
              <span>{location ? `${location.city}, ${location.country}` : 'Set Location'}</span>
            </button>

            {/* AI Brief */}
            {user && (
              <button
                onClick={onShowAIBrief}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 transition-all text-sm"
              >
                <Bot size={16} />
                <span>AI Brief</span>
              </button>
            )}

            {/* User Menu */}
            {user ? (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-lg">
                  <User size={16} />
                  <span className="text-sm">{user.name}</span>
                </div>
                <button
                  onClick={logout}
                  className="p-2 text-gray-600 hover:text-red-600 transition-colors"
                  title="Logout"
                >
                  <LogOut size={16} />
                </button>
              </div>
            ) : (
              <button
                onClick={onShowAuth}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
              >
                Login
              </button>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setShowMobileMenu(!showMobileMenu)}
            className="md:hidden p-2 text-gray-600"
          >
            {showMobileMenu ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {showMobileMenu && (
          <div className="md:hidden py-4 border-t border-gray-200 space-y-2">
            <button
              onClick={() => {
                onShowLocation();
                setShowMobileMenu(false);
              }}
              className="flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors text-sm"
            >
              <MapPin size={16} />
              <span>{location ? `${location.city}, ${location.country}` : 'Set Location'}</span>
            </button>

            {user && (
              <button
                onClick={() => {
                  onShowAIBrief();
                  setShowMobileMenu(false);
                }}
                className="flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm"
              >
                <Bot size={16} />
                <span>AI Brief</span>
              </button>
            )}

            {user ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-lg">
                  <User size={16} />
                  <span className="text-sm">{user.name}</span>
                </div>
                <button
                  onClick={() => {
                    logout();
                    setShowMobileMenu(false);
                  }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-sm"
                >
                  <LogOut size={16} />
                  <span>Logout</span>
                </button>
              </div>
            ) : (
              <button
                onClick={() => {
                  onShowAuth();
                  setShowMobileMenu(false);
                }}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
              >
                Login
              </button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
