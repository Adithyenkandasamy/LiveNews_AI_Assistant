import { NextRequest, NextResponse } from 'next/server';

const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const location = searchParams.get('location');
  const category = searchParams.get('category');
  const country = searchParams.get('country');
  const city = searchParams.get('city');
  const tags = searchParams.get('tags');

  try {
    // Proxy request to Flask backend
    const flaskUrl = new URL('/api/news', FLASK_API_URL);
    if (location) flaskUrl.searchParams.set('location', location);
    if (category) flaskUrl.searchParams.set('category', category);
    if (country) flaskUrl.searchParams.set('country', country);
    if (city) flaskUrl.searchParams.set('city', city);
    if (tags) flaskUrl.searchParams.set('tags', tags);

    const response = await fetch(flaskUrl.toString());
    
    if (!response.ok) {
      throw new Error(`Flask API error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('Error connecting to Flask backend:', error);
    
    // Fallback to mock data if Flask backend is unavailable
    const mockArticles = [
    {
      id: '1',
      title: 'Mumbai Tech Hub Announces Major AI Research Center',
      summary: 'New artificial intelligence research facility to create 5,000 jobs in Mumbai, focusing on healthcare AI solutions...',
      content: 'Mumbai is set to become a major AI research hub with the announcement of a new ₹2,000 crore facility in Bandra-Kurla Complex. The center will focus on developing AI solutions for healthcare, finance, and urban planning...',
      source: 'Mumbai Mirror',
      publishedAt: '2024-01-15T10:30:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400',
      category: 'technology',
      tags: ['AI', 'Mumbai', 'Healthcare', 'Jobs', 'Research'],
      sentiment: 'positive',
      location: { country: 'IN', city: 'Mumbai' },
      readTime: 4,
      aiSummary: 'Mumbai is establishing itself as an AI research hub with a new ₹2,000 crore facility that will create thousands of jobs and focus on healthcare AI solutions.'
    },
    {
      id: '2',
      title: 'Pakistan defence minister Khawaja Asif fumbles 7 times at UNSC: "Vital to risks"',
      summary: 'Pakistan minister Khawaja Asif was speaking at the United Nation Security Council\'s AI Innovation Dialogue.',
      content: 'Pakistan minister Khawaja Asif was speaking at the United Nation Security Council\'s AI Innovation Dialogue. The minister discussed various aspects of AI governance and international cooperation in the field of artificial intelligence.',
      source: 'Hindustan Times',
      category: 'Technology',
      publishedAt: '2025-09-26T11:09:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&h=200&fit=crop',
      url: 'https://example.com/news/1',
      readingTime: 1,
      sentiment: -0.2,
      keywords: ['Pakistan', 'UNSC', 'AI', 'Technology']
    },
    {
      id: '3',
      title: 'AI Revolution Transforms Global Industries',
      summary: 'Artificial Intelligence continues to reshape multiple sectors including healthcare, finance, and transportation with groundbreaking innovations.',
      content: 'The AI revolution is fundamentally transforming how industries operate across the globe. From healthcare diagnostics to financial services and autonomous transportation, artificial intelligence is creating new possibilities and efficiencies.',
      source: 'Tech Today',
      category: 'Technology',
      publishedAt: '2025-09-26T10:30:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400&h=200&fit=crop',
      url: 'https://example.com/news/2',
      readingTime: 3,
      sentiment: 0.8,
      keywords: ['AI', 'Technology', 'Innovation', 'Industries']
    },
    {
      id: '4',
      title: 'Climate Summit Reaches Breakthrough Agreement',
      summary: 'World leaders announce ambitious new targets for carbon reduction and renewable energy adoption at the latest climate summit.',
      content: 'In a historic moment for environmental policy, world leaders have reached a breakthrough agreement on climate action, setting ambitious new targets for carbon reduction and renewable energy adoption.',
      source: 'Global News',
      category: 'Environment',
      publishedAt: '2025-09-26T09:15:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1569163139394-de4e4f43e4e3?w=400&h=200&fit=crop',
      url: 'https://example.com/news/3',
      readingTime: 4,
      sentiment: 0.6,
      keywords: ['Climate', 'Environment', 'Renewable Energy']
    },
    {
      id: '5',
      title: `Local News from ${city || country}`,
      summary: `Breaking developments and regional updates specifically relevant to ${city || country}.`,
      content: `This article covers the latest developments and news specifically relevant to ${city || country}, providing localized insights and coverage.`,
      source: 'Regional Times',
      category: 'Local',
      publishedAt: '2025-09-26T08:00:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&h=200&fit=crop',
      url: 'https://example.com/news/4',
      readingTime: 2,
      sentiment: 0.3,
      keywords: ['Local', country, city].filter(Boolean)
    },
    {
      id: '6',
      title: 'Global Market Updates Show Positive Trends',
      summary: 'International markets demonstrate resilience with technology and renewable energy sectors leading growth.',
      content: 'Global markets are showing positive trends with technology and renewable energy sectors leading the charge. Investors are increasingly confident in sustainable growth strategies.',
      source: 'Financial Wire',
      category: 'Business',
      publishedAt: '2025-09-26T07:30:00Z',
      imageUrl: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=400&h=200&fit=crop',
      url: 'https://example.com/news/5',
      readingTime: 3,
      sentiment: 0.7,
      keywords: ['Markets', 'Finance', 'Technology', 'Investment']
    }
    ];

    // Filter by location if specified
    let filteredArticles = mockArticles;
    
    if (country) {
      filteredArticles = filteredArticles.filter(article => 
        article.location?.country === country || 
        article.tags?.some(tag => tag.toLowerCase().includes(country.toLowerCase()))
      );
    }
    
    if (city) {
      filteredArticles = filteredArticles.filter(article => 
        article.location?.city === city || 
        article.tags?.some(tag => tag.toLowerCase().includes(city.toLowerCase()))
      );
    }
    
    // Filter by category if specified
    if (category && category !== 'all') {
      filteredArticles = filteredArticles.filter(article => 
        article.category.toLowerCase() === category.toLowerCase()
      );
    }
    
    // Filter by tags if specified
    if (tags) {
      const tagList = tags.split(',').map(tag => tag.trim().toLowerCase());
      filteredArticles = filteredArticles.filter(article => 
        article.tags?.some(tag => 
          tagList.some(searchTag => tag.toLowerCase().includes(searchTag))
        )
      );
    }

    return NextResponse.json({
      articles: filteredArticles,
      total: filteredArticles.length,
      location: { country, city },
      availableTags: [...new Set(mockArticles.flatMap(article => article.tags || []))],
      categories: [...new Set(mockArticles.map(article => article.category))]
    });
  }
}
