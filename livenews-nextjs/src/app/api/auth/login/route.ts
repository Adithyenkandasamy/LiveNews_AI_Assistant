import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json();

    // For demo purposes - replace with actual authentication logic
    // In production, verify credentials against database
    if (email && password) {
      const mockUser = {
        id: '1',
        name: email.split('@')[0],
        email: email,
        location: 'New York',
        preferences: {
          categories: ['Technology', 'Business', 'Environment'],
          sources: ['TechCrunch', 'Reuters', 'BBC'],
          aiSummary: true
        }
      };

      return NextResponse.json({
        success: true,
        user: mockUser,
        message: 'Login successful'
      });
    }

    return NextResponse.json(
      { success: false, error: 'Invalid credentials' },
      { status: 401 }
    );

  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Login failed' },
      { status: 500 }
    );
  }
}
