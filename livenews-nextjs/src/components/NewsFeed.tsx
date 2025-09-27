'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MapPin, Clock, Share2, Bookmark, ExternalLink, TrendingUp, MessageCircle } from 'lucide-react';
import { TagFilter } from './TagFilter';

interface Article {
  id: string;
  title: string;
  summary: string;
  content?: string;
  source: string;
  category: string;
  publishedAt: string;
  imageUrl?: string;
  url: string;
  keywords?: string[];
}

interface NewsFeedProps {
  location: any;
  user: any;
  onArticleSelect: (article: Article) => void;
  onShare: (article: Article) => void;
}

export function NewsFeed({ location, user, onArticleSelect, onShare }: NewsFeedProps) {
  const [articles, setArticles] = useState<Article[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const staticCategories = ['all', 'technology', 'business', 'sports', 'entertainment', 'health'];

  useEffect(() => {
    fetchNews();
  }, [location, user]);

  const fetchNews = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      
      if (location) {
        params.append('country', location.country);
        params.append('city', location.city);
      }
      
      if (user?.preferences?.categories?.length) {
        params.append('categories', user.preferences.categories.join(','));
      }

      const response = await fetch(`/api/news?${params}`);
      if (response.ok) {
        const data = await response.json();
        setArticles(data.articles || []);
      }
    } catch (error) {
      console.error('Failed to fetch news:', error);
      // Load demo articles for development
      setArticles(getDemoArticles());
    } finally {
      setLoading(false);
    }
  };

  const getDemoArticles = (): Article[] => [
    {
      id: '1',
      title: 'Pakistan defence minister Khawaja Asif fumbles 7 times at UNSC: "Vital to risks"',
      summary: 'Pakistan minister Khawaja Asif was speaking at the United Nation Security Council\'s AI Innovation Dialogue.',
      source: 'Hindustan Times',
      category: 'Technology',
      publishedAt: '2025-09-26T11:09:00Z',
      imageUrl: '/api/placeholder/400/200',
      url: 'https://example.com/news/1',
      keywords: ['Pakistan', 'UNSC', 'AI', 'Technology']
    },
    {
      id: '2',
      title: 'AI Revolution Transforms Global Industries',
      summary: 'Artificial Intelligence continues to reshape multiple sectors including healthcare, finance, and transportation with groundbreaking innovations.',
      source: 'Tech Today',
      category: 'Technology',
      publishedAt: '2025-09-26T10:30:00Z',
      imageUrl: '/api/placeholder/400/200',
      url: 'https://example.com/news/2',
      keywords: ['AI', 'Technology', 'Innovation', 'Industries']
    },
    {
      id: '3',
      title: 'Climate Summit Reaches Breakthrough Agreement',
      summary: 'World leaders announce ambitious new targets for carbon reduction and renewable energy adoption at the latest climate summit.',
      source: 'Global News',
      category: 'Environment',
      publishedAt: '2025-09-26T09:15:00Z',
      imageUrl: '/api/placeholder/400/200',
      url: 'https://example.com/news/3',
      keywords: ['Climate', 'Environment', 'Renewable Energy']
    }
  ];

  const filteredArticles = articles.filter(article => {
    const matchesFilter = filter === 'all' || article.category.toLowerCase() === filter;
    const matchesSearch = searchQuery === '' || 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.summary.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const dynamicCategories = ['all', ...Array.from(new Set(articles.map(a => a.category.toLowerCase())))];

  if (loading) {
    return (
      <div className="space-y-6">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/20 animate-pulse">
            <div className="h-4 bg-gray-200 rounded mb-2"></div>
            <div className="h-3 bg-gray-200 rounded mb-4 w-3/4"></div>
            <div className="h-32 bg-gray-200 rounded mb-4"></div>
            <div className="flex gap-2">
              <div className="h-6 w-16 bg-gray-200 rounded"></div>
              <div className="h-6 w-20 bg-gray-200 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="glass-morphism rounded-2xl p-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <input
              type="text"
              placeholder="🔍 Search news..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-5 py-3 pl-12 bg-white/80 backdrop-blur border border-gray-200/50 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all placeholder-gray-500"
            />
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400">
              🔍
            </div>
          </div>
          <div className="flex gap-2 overflow-x-auto scrollbar-hide">
            {dynamicCategories.map(category => (
              <button
                key={category}
                onClick={() => setFilter(category)}
                className={`px-5 py-3 rounded-xl text-sm font-medium whitespace-nowrap transition-all ${
                  filter === category
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg scale-105'
                    : 'bg-white/60 text-gray-700 hover:bg-white/80 border border-gray-200/50'
                }`}
              >
                {category === 'all' ? '✨ All News' : '📰 ' + category.charAt(0).toUpperCase() + category.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Location-based News Header */}
      {location && (
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl p-6 animate-gradient"
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            backgroundSize: '200% 200%',
          }}
        >
          <div className="absolute inset-0 bg-white/10 backdrop-blur-sm"></div>
          <div className="relative z-10 text-white">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-white/20 rounded-lg backdrop-blur">
                <MapPin size={24} />
              </div>
              <div>
                <h2 className="text-xl font-bold">📍 {location.city}, {location.country}</h2>
                <p className="text-purple-100 text-sm">Local & Personalized Feed</p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm">
              <span className="flex items-center gap-2">
                <TrendingUp size={16} />
                {filteredArticles.length} Articles
              </span>
              <span className="flex items-center gap-2">
                <Clock size={16} />
                Live Updates
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Articles */}
      <div className="space-y-4">
        {filteredArticles.map((article, index) => (
          <motion.article
            key={article.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass-morphism rounded-2xl p-6 news-card-hover card-shine cursor-pointer group"
            onClick={() => onArticleSelect(article)}
          >
            <div className="flex flex-col lg:flex-row gap-4">
              {/* Article Image */}
              {article.imageUrl && (
                <div className="lg:w-56 h-40 lg:h-32 rounded-xl overflow-hidden flex-shrink-0 bg-gradient-to-br from-gray-100 to-gray-200">
                  <img
                    src={article.imageUrl}
                    alt={article.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                </div>
              )}

              {/* Article Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
                    {article.title}
                  </h3>
                  
                  {/* Actions */}
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onShare(article);
                      }}
                      className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                      title="Share"
                    >
                      <Share2 size={16} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Bookmark functionality
                      }}
                      className="p-2 text-gray-500 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                      title="Bookmark"
                    >
                      <Bookmark size={16} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        window.open(article.url, '_blank');
                      }}
                      className="p-2 text-gray-500 hover:text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                      title="Open Original"
                    >
                      <ExternalLink size={16} />
                    </button>
                  </div>
                </div>

                <p className="text-gray-600 text-sm mb-3 line-clamp-2">
                  {article.summary}
                </p>

                {/* Article Meta */}
                <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500">
                  <span className="font-medium text-blue-600">{article.source}</span>
                  <span className="px-2 py-1 bg-gray-100 rounded-full">{article.category}</span>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <Clock size={14} />
                      <span>{new Date(article.publishedAt).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>

                {/* Keywords */}
                {article.keywords && article.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-3">
                    {article.keywords.slice(0, 4).map(keyword => (
                      <span
                        key={keyword}
                        className="px-2 py-1 bg-blue-50 text-blue-600 text-xs rounded-full"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.article>
        ))}
      </div>

      {filteredArticles.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg mb-2">📰</div>
          <h3 className="text-lg font-medium text-gray-600 mb-2">No articles found</h3>
          <p className="text-gray-500 text-sm">
            Try adjusting your filters or search terms
          </p>
        </div>
      )}
    </div>
  );
}
