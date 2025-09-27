import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { preferences } = await request.json();

    // For demo purposes - replace with actual user preference update logic
    // In production, update user preferences in database
    
    return NextResponse.json({
      success: true,
      preferences: preferences,
      message: 'Preferences updated successfully'
    });

  } catch (error) {
    console.error('Error updating preferences:', error);
    return NextResponse.json(
      { error: 'Failed to update preferences' },
      { status: 500 }
    );
  }
}
