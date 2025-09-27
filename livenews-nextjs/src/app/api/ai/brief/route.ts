import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { location, preferences } = await request.json();

    // For demo purposes, return a comprehensive AI brief
    // In production, integrate with Gemini AI for real analysis
    const mockBrief = {
      summary: `Today's key developments include significant AI policy discussions at the UN Security Council, breakthrough climate agreements, and major technological advancements${location ? ` specifically relevant to ${location.city}, ${location.country}` : ''}. The Pakistan defense minister's remarks on AI governance have sparked international dialogue about artificial intelligence regulation and its global implications.`,
      
      sections: [
        {
          title: 'AI & Technology',
          content: `Pakistan defense minister Khawaja Asif addressed the UN Security Council on AI governance, highlighting the critical need for international cooperation in AI regulation. The discussion emphasized balancing innovation with ethical considerations and national security implications${location ? ` with specific relevance to ${location.country}'s tech sector` : ''}.`,
          category: 'Technology',
          importance: 'high' as const,
          articles: 8
        },
        {
          title: 'Global Politics',
          content: `International relations continue to evolve with new diplomatic initiatives and policy frameworks. Recent discussions at various international forums indicate a shift towards more collaborative approaches to global challenges${location ? `, with ${location.country} playing an active role in regional diplomacy` : ''}.`,
          category: 'Politics',
          importance: 'high' as const,
          articles: 12
        },
        {
          title: 'Climate & Environment',
          content: `Environmental initiatives gain momentum with new renewable energy projects and sustainability commitments from major corporations. Climate summit agreements show promising progress toward carbon neutrality goals${location ? `, with ${location.city} leading regional sustainability efforts` : ''}.`,
          category: 'Environment',
          importance: 'medium' as const,
          articles: 6
        },
        {
          title: 'Economic Updates',
          content: `Market trends reflect growing confidence in tech sectors and renewable energy investments. Global economic indicators suggest steady recovery with emerging markets showing particular strength${location ? `, including positive trends in ${location.country}'s economy` : ''}.`,
          category: 'Business',
          importance: 'medium' as const,
          articles: 10
        }
      ],

      insights: [
        'AI governance is becoming a critical diplomatic priority',
        'Climate commitments are accelerating across industries',
        'Technology regulation requires international coordination',
        'Economic recovery shows regional variations',
        ...(location ? [`${location.country} is actively engaged in global policy discussions`] : [])
      ],

      trending: [
        'AI Regulation',
        'Climate Action',
        'International Diplomacy',
        'Tech Innovation',
        'Sustainable Energy',
        ...(location ? [`${location.city} News`] : [])
      ],

      generated_at: new Date().toISOString(),
      
      location_specific: location ? {
        local_headlines: [
          `${location.city} leads regional sustainability initiatives`,
          `Local tech sector shows strong growth in ${location.country}`,
          `${location.city} participates in global climate summit`
        ],
        weather_context: `Current conditions in ${location.city} support outdoor activities`,
        regional_trends: [`Growing tech investment in ${location.country}`, `Renewable energy adoption in ${location.city}`]
      } : null
    };

    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    return NextResponse.json(mockBrief);

  } catch (error) {
    console.error('Error generating AI brief:', error);
    return NextResponse.json(
      { error: 'Failed to generate AI brief' },
      { status: 500 }
    );
  }
}
