'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/providers/AuthProvider';
import { useLocation } from '@/components/providers/LocationProvider';
import { Header } from '@/components/Header';
import { NewsFeed } from '@/components/NewsFeed';
import { AuthModal } from '@/components/AuthModal';
import { LocationSelector } from '@/components/LocationSelector';
import { AIBriefPanel } from '@/components/AIBriefPanel';
import { ShareModal } from '@/components/ShareModal';
import { RAGChat } from '@/components/RAGChat';

export default function Home() {
  const { user } = useAuth();
  const { location } = useLocation();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showLocationSelector, setShowLocationSelector] = useState(false);
  const [showAIBrief, setShowAIBrief] = useState(false);
  const [showRAGChat, setShowRAGChat] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState<any>(null);
  const [shareArticle, setShareArticle] = useState<any>(null);

  useEffect(() => {
    // Show auth modal if user is not logged in after 3 seconds
    if (!user) {
      const timer = setTimeout(() => setShowAuthModal(true), 3000);
      return () => clearTimeout(timer);
    }
  }, [user]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Header 
        onShowAuth={() => setShowAuthModal(true)}
        onShowLocation={() => setShowLocationSelector(true)}
        onShowAIBrief={() => setShowAIBrief(true)}
      />
      
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main News Feed */}
          <div className="lg:col-span-2">
            <NewsFeed 
              location={location}
              user={user}
              onArticleSelect={setSelectedArticle}
              onShare={setShareArticle}
            />
          </div>
          
          {/* Side Panel */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 space-y-6">
              {/* Location Info */}
              {location && (
                <div className="glass-morphism rounded-2xl p-5">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 bg-gradient-to-br from-green-500 to-blue-500 rounded-xl text-white">
                      📍
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-gray-800">Your Location</h3>
                      <p className="text-sm text-gray-600">{location.city}, {location.country}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowLocationSelector(true)}
                    className="w-full mt-2 px-4 py-2 bg-white/50 hover:bg-white/70 rounded-xl text-sm font-medium text-blue-600 transition-all"
                  >
                    Change Location
                  </button>
                </div>
              )}
              
              {/* AI Brief */}
              {user && (
                <div className="glass-morphism rounded-2xl p-5">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl text-white animate-pulse-glow">
                      🤖
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-800">AI News Brief</h3>
                      <p className="text-xs text-gray-500">Powered by Gemini AI</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    Get personalized AI-powered summaries and insights from your news feed.
                  </p>
                  <button
                    onClick={() => setShowAIBrief(true)}
                    className="btn-primary w-full"
                  >
                    ✨ Generate AI Brief
                  </button>
                </div>
              )}
              
              {/* Trending Topics */}
              <div className="glass-morphism rounded-2xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl animate-float">🔥</span>
                  <h3 className="font-bold text-gray-800">Trending Now</h3>
                </div>
                <div className="space-y-3">
                  {[
                    { topic: 'AI Technology', emoji: '🤖' },
                    { topic: 'Climate Change', emoji: '🌍' },
                    { topic: 'Global Politics', emoji: '🌐' },
                    { topic: 'Sports News', emoji: '⚽' },
                    { topic: 'Market Updates', emoji: '📈' }
                  ].map((item, index) => (
                    <div 
                      key={index} 
                      className="flex items-center justify-between p-2 hover:bg-white/50 rounded-lg transition-all cursor-pointer group"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg group-hover:scale-110 transition-transform">{item.emoji}</span>
                        <span className="text-sm font-medium text-gray-700 group-hover:text-blue-600">{item.topic}</span>
                      </div>
                      <span className="px-2 py-1 bg-gradient-to-r from-blue-500 to-purple-500 text-white text-xs rounded-full">
                        #{index + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* AI Chat Button */}
              {user && (
                <div className="glass-morphism rounded-2xl p-5">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-gradient-to-br from-green-500 to-teal-500 rounded-xl text-white animate-pulse-glow">
                      💬
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-800">Ask Questions</h3>
                      <p className="text-xs text-gray-500">RAG-powered Q&A</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    Ask questions about any news article or topic using our AI assistant.
                  </p>
                  <button
                    onClick={() => setShowRAGChat(true)}
                    className="btn-primary w-full"
                  >
                    🤖 Ask AI Assistant
                  </button>
                </div>
              )}

              {/* Quick Stats */}
              {user && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="glass-morphism rounded-xl p-4 text-center">
                    <div className="text-2xl mb-1">📰</div>
                    <div className="text-2xl font-bold text-gradient">156</div>
                    <div className="text-xs text-gray-600">Articles Read</div>
                  </div>
                  <div className="glass-morphism rounded-xl p-4 text-center">
                    <div className="text-2xl mb-1">⭐</div>
                    <div className="text-2xl font-bold text-gradient">42</div>
                    <div className="text-xs text-gray-600">Saved Articles</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Modals */}
      {showAuthModal && (
        <AuthModal onClose={() => setShowAuthModal(false)} />
      )}
      
      {showLocationSelector && (
        <LocationSelector onClose={() => setShowLocationSelector(false)} />
      )}
      
      {showAIBrief && (
        <AIBriefPanel 
          location={location}
          user={user}
          onClose={() => setShowAIBrief(false)} 
        />
      )}
      
      {shareArticle && (
        <ShareModal 
          article={shareArticle} 
          onClose={() => setShareArticle(null)} 
        />
      )}

      {showRAGChat && (
        <RAGChat 
          isOpen={showRAGChat}
          onClose={() => setShowRAGChat(false)} 
        />
      )}
    </div>
  );
}
