'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Tag, X } from 'lucide-react';

interface TagFilterProps {
  selectedTags: string[];
  onTagsChange: (tags: string[]) => void;
  availableTags: string[];
}

export function TagFilter({ selectedTags, onTagsChange, availableTags }: TagFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      onTagsChange(selectedTags.filter(t => t !== tag));
    } else {
      onTagsChange([...selectedTags, tag]);
    }
  };

  const clearAllTags = () => {
    onTagsChange([]);
  };

  const popularTags = ['AI', 'Technology', 'Healthcare', 'Climate', 'Business', 'Politics'];
  const displayTags = isExpanded ? availableTags : popularTags.filter(tag => availableTags.includes(tag));

  return (
    <div className="glass-morphism rounded-2xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Tag size={20} className="text-purple-600" />
          <h3 className="font-bold text-gray-800">Filter by Tags</h3>
        </div>
        {selectedTags.length > 0 && (
          <button
            onClick={clearAllTags}
            className="text-sm text-red-600 hover:text-red-800 transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Selected Tags */}
      {selectedTags.length > 0 && (
        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-2">Selected:</p>
          <div className="flex flex-wrap gap-2">
            {selectedTags.map((tag) => (
              <motion.span
                key={tag}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex items-center gap-1 px-3 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm rounded-full"
              >
                {tag}
                <button
                  onClick={() => toggleTag(tag)}
                  className="hover:bg-white/20 rounded-full p-0.5 transition-colors"
                >
                  <X size={12} />
                </button>
              </motion.span>
            ))}
          </div>
        </div>
      )}

      {/* Available Tags */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {displayTags.map((tag) => (
            <button
              key={tag}
              onClick={() => toggleTag(tag)}
              className={`px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                selectedTags.includes(tag)
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg scale-105'
                  : 'bg-white/60 text-gray-700 hover:bg-white/80 border border-gray-200/50'
              }`}
            >
              #{tag}
            </button>
          ))}
        </div>

        {availableTags.length > popularTags.length && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            {isExpanded ? 'Show Less' : `Show All ${availableTags.length} Tags`}
          </button>
        )}
      </div>
    </div>
  );
}
