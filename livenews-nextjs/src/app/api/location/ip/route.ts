import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // Get client IP address
    const forwarded = request.headers.get('x-forwarded-for');
    const ip = forwarded ? forwarded.split(',')[0] : request.headers.get('x-real-ip') || '127.0.0.1';

    // For demo purposes, return mock location data based on IP
    // In production, use a geolocation service like ipapi.co or MaxMind
    const mockLocationData = {
      ip: ip,
      country: 'IN',
      country_name: 'India',
      city: 'Mumbai',
      region: 'Maharashtra',
      timezone: 'Asia/Kolkata',
      latitude: 19.0760,
      longitude: 72.8777
    };

    return NextResponse.json({
      country: mockLocationData.country,
      city: mockLocationData.city,
      region: mockLocationData.region,
      coordinates: {
        latitude: mockLocationData.latitude,
        longitude: mockLocationData.longitude
      },
      timezone: mockLocationData.timezone
    });

  } catch (error) {
    console.error('Error getting IP location:', error);
    return NextResponse.json(
      { error: 'Failed to get location from IP' },
      { status: 500 }
    );
  }
}
