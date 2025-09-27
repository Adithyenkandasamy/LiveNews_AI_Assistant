'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { X } from 'lucide-react';

interface Toast {
  id: string;
  title?: string;
  description: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface ToastContextType {
  toasts: Toast[];
  toast: (toast: Omit<Toast, 'id'>) => void;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = (newToast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const toastWithId = { ...newToast, id };
    
    setToasts(prev => [...prev, toastWithId]);
    
    // Auto dismiss after duration
    setTimeout(() => {
      dismiss(id);
    }, newToast.duration || 5000);
  };

  const dismiss = (id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ toasts, toast, dismiss }}>
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`
              p-4 rounded-lg shadow-lg max-w-sm backdrop-blur-sm border
              ${toast.type === 'success' ? 'bg-green-500/90 text-white border-green-400' : ''}
              ${toast.type === 'error' ? 'bg-red-500/90 text-white border-red-400' : ''}
              ${toast.type === 'info' ? 'bg-blue-500/90 text-white border-blue-400' : ''}
              ${toast.type === 'warning' ? 'bg-yellow-500/90 text-white border-yellow-400' : ''}
              animate-slide-in-bottom
            `}
          >
            <div className="flex items-start gap-2">
              <div className="flex-1">
                {toast.title && (
                  <div className="font-semibold mb-1">{toast.title}</div>
                )}
                <div className="text-sm">{toast.description}</div>
              </div>
              <button
                onClick={() => dismiss(toast.id)}
                className="text-white/80 hover:text-white transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (context === undefined) {
    throw new Error('useToast must be used within a Toaster');
  }
  return context;
}
