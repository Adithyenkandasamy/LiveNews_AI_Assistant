import { NextResponse } from 'next/server';

const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function POST(request: Request) {
  try {
    const { query, context } = await request.json();

    // Try to connect to Flask backend first
    try {
      const response = await fetch(`${FLASK_API_URL}/api/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, context }),
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      console.error('Error connecting to Flask backend:', error);
    }

    // Fallback to mock RAG response if Flask backend is unavailable
    const mockResponses = {
      'ai': {
        answer: 'Based on recent news articles, AI technology is rapidly advancing across multiple sectors. Mumbai is establishing a new ₹2,000 crore AI research center focusing on healthcare solutions, which will create 5,000 jobs. Globally, AI is transforming industries including healthcare diagnostics, financial services, and autonomous transportation.',
        sources: ['Mumbai Tech Hub Announces Major AI Research Center', 'AI Revolution Transforms Global Industries']
      },
      'mumbai': {
        answer: 'Mumbai is experiencing significant technological growth with the announcement of a major AI research center in Bandra-Kurla Complex. This ₹2,000 crore facility will focus on developing AI solutions for healthcare, finance, and urban planning, creating approximately 5,000 new jobs in the tech sector.',
        sources: ['Mumbai Tech Hub Announces Major AI Research Center']
      },
      'climate': {
        answer: 'Recent climate news shows positive developments with world leaders reaching a breakthrough agreement at the latest climate summit. The agreement sets ambitious new targets for carbon reduction and renewable energy adoption, representing a historic moment for environmental policy.',
        sources: ['Climate Summit Reaches Breakthrough Agreement']
      },
      'market': {
        answer: 'Global markets are showing positive trends with technology and renewable energy sectors leading growth. International markets demonstrate resilience, and investors are increasingly confident in sustainable growth strategies.',
        sources: ['Global Market Updates Show Positive Trends']
      }
    };

    // Simple keyword matching for demo
    const queryLower = query.toLowerCase();
    let response: { answer: string; sources: string[] } = {
      answer: "I couldn't find specific information about that topic in the current news articles. Could you try asking about AI, technology, climate, markets, or Mumbai-related news?",
      sources: []
    };

    // Check for keywords and return relevant response
    for (const [keyword, mockResponse] of Object.entries(mockResponses)) {
      if (queryLower.includes(keyword)) {
        response = mockResponse;
        break;
      }
    }

    // Add some delay to simulate processing
    await new Promise(resolve => setTimeout(resolve, 1000));

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error processing RAG query:', error);
    return NextResponse.json(
      { error: 'Failed to process query' },
      { status: 500 }
    );
  }
}
