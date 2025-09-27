'use client';

import { useState } from 'react';
import { X, Share2, Copy, Twitter, Facebook, Linkedin, Mail, Link, QrCode } from 'lucide-react';
import { motion } from 'framer-motion';

interface ShareModalProps {
  article: any;
  onClose: () => void;
}

export function ShareModal({ article, onClose }: ShareModalProps) {
  const [copied, setCopied] = useState(false);
  const [showQR, setShowQR] = useState(false);

  // Generate shareable URL that redirects to our website
  const shareUrl = `https://livenews-ai.com/share/${article.id}`;
  const shareText = `${article.title} - Read this personalized news article with AI insights on LiveNews AI`;

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const shareOptions = [
    {
      name: 'Copy Link',
      icon: Copy,
      color: 'bg-gray-600',
      action: copyToClipboard,
    },
    {
      name: 'Twitter',
      icon: Twitter,
      color: 'bg-blue-500',
      action: () => window.open(
        `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`,
        '_blank'
      ),
    },
    {
      name: 'Facebook',
      icon: Facebook,
      color: 'bg-blue-600',
      action: () => window.open(
        `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`,
        '_blank'
      ),
    },
    {
      name: 'LinkedIn',
      icon: Linkedin,
      color: 'bg-blue-700',
      action: () => window.open(
        `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`,
        '_blank'
      ),
    },
    {
      name: 'Email',
      icon: Mail,
      color: 'bg-red-600',
      action: () => window.open(
        `mailto:?subject=${encodeURIComponent(article.title)}&body=${encodeURIComponent(shareText + '\n\n' + shareUrl)}`,
        '_blank'
      ),
    },
    {
      name: 'QR Code',
      icon: QrCode,
      color: 'bg-purple-600',
      action: () => setShowQR(true),
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Share2 size={24} />
              <div>
                <h2 className="text-lg font-bold">Share Article</h2>
                <p className="text-indigo-100 text-sm">Share via LiveNews AI platform</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-white/80 hover:text-white hover:bg-white/20 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="p-6">
          {/* Article Preview */}
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-2 line-clamp-2">
              {article.title}
            </h3>
            <p className="text-sm text-gray-600 line-clamp-2">
              {article.summary}
            </p>
            <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
              <span>{article.source}</span>
              <span>•</span>
              <span>{article.category}</span>
              {article.readingTime && (
                <>
                  <span>•</span>
                  <span>{article.readingTime} min read</span>
                </>
              )}
            </div>
          </div>

          {/* Share URL */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Share Link (redirects to LiveNews AI)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={shareUrl}
                readOnly
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-sm"
              />
              <button
                onClick={copyToClipboard}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  copied
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-600 text-white hover:bg-gray-700'
                }`}
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>

          {/* Share Options */}
          <div className="space-y-3">
            <h4 className="font-medium text-gray-800">Share Options</h4>
            <div className="grid grid-cols-2 gap-3">
              {shareOptions.map((option) => (
                <motion.button
                  key={option.name}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={option.action}
                  className={`flex items-center gap-3 p-3 ${option.color} text-white rounded-xl hover:opacity-90 transition-opacity`}
                >
                  <option.icon size={20} />
                  <span className="font-medium">{option.name}</span>
                </motion.button>
              ))}
            </div>
          </div>

          {/* LiveNews AI Branding */}
          <div className="mt-6 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <div className="w-6 h-6 bg-gradient-to-br from-blue-600 to-purple-600 rounded flex items-center justify-center">
                <span className="text-white font-bold text-xs">LN</span>
              </div>
              <span>Shared via LiveNews AI - Personalized Intelligence</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Recipients will be directed to LiveNews AI to read the full article with AI-powered insights and personalized recommendations.
            </p>
          </div>

          {/* Features Highlight */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-700 font-medium mb-2">Why share via LiveNews AI?</p>
            <div className="grid grid-cols-2 gap-1 text-xs text-blue-600">
              <div>✓ AI-powered summaries</div>
              <div>✓ Fact-checking insights</div>
              <div>✓ Related article suggestions</div>
              <div>✓ Personalized experience</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* QR Code Modal */}
      {showQR && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl p-6 mx-4 max-w-sm w-full"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">QR Code</h3>
              <button
                onClick={() => setShowQR(false)}
                className="p-1 text-gray-500 hover:text-gray-700"
              >
                <X size={20} />
              </button>
            </div>
            
            <div className="text-center">
              <div className="w-48 h-48 bg-gray-100 rounded-lg mx-auto mb-4 flex items-center justify-center">
                {/* QR Code would be generated here - using placeholder */}
                <div className="text-gray-400">
                  <QrCode size={48} />
                  <p className="text-sm mt-2">QR Code for<br />{shareUrl}</p>
                </div>
              </div>
              <p className="text-sm text-gray-600">
                Scan to open article in LiveNews AI
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
