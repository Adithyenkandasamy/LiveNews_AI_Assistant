import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const lat = searchParams.get('lat');
    const lng = searchParams.get('lng');

    if (!lat || !lng) {
      return NextResponse.json(
        { error: 'Latitude and longitude are required' },
        { status: 400 }
      );
    }

    // For demo purposes, return mock reverse geocoding data
    // In production, use Google Maps Geocoding API or similar service
    const mockLocationData = {
      country: 'IN',
      city: 'Mumbai',
      region: 'Maharashtra',
      address: 'Mumbai, Maharashtra, India',
      postal_code: '400001'
    };

    return NextResponse.json(mockLocationData);

  } catch (error) {
    console.error('Error with reverse geocoding:', error);
    return NextResponse.json(
      { error: 'Failed to reverse geocode location' },
      { status: 500 }
    );
  }
}
