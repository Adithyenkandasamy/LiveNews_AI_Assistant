import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { name, email, password } = await request.json();

    // Validate input
    if (!name || !email || !password) {
      return NextResponse.json(
        { success: false, error: 'All fields are required' },
        { status: 400 }
      );
    }

    if (password.length < 6) {
      return NextResponse.json(
        { success: false, error: 'Password must be at least 6 characters' },
        { status: 400 }
      );
    }

    // For demo purposes - replace with actual user creation logic
    // In production, hash password and save to database
    const newUser = {
      id: Date.now().toString(),
      name: name,
      email: email,
      location: 'New York',
      preferences: {
        categories: ['Technology', 'Business'],
        sources: ['TechCrunch', 'Reuters'],
        aiSummary: true
      }
    };

    return NextResponse.json({
      success: true,
      user: newUser,
      message: 'Account created successfully'
    });

  } catch (error) {
    console.error('Signup error:', error);
    return NextResponse.json(
      { error: 'Signup failed' },
      { status: 500 }
    );
  }
}
