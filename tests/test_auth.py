import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import User
from sqlmodel import select, Session, create_engine, SQLModel
import os

# Use a test database
TEST_DB = "test_chat.db"


@pytest.fixture(autouse=True)
def setup_db():
    """Setup test database before each test"""
    # Remove existing test db
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    
    # Create new test engine and database
    engine = create_engine(f"sqlite:///{TEST_DB}", echo=False, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    
    # Update the app to use test engine
    import app.db as db_module
    old_engine = db_module.engine
    db_module.engine = engine
    
    # Re-create get_session to use the test engine
    def get_test_session():
        with Session(engine) as session:
            yield session
    
    # Override dependency
    from app.main import get_session as app_get_session
    app.dependency_overrides[app_get_session] = get_test_session
    
    yield engine
    
    # Restore original engine
    db_module.engine = old_engine
    app.dependency_overrides.clear()
    
    # Cleanup after test
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


client = TestClient(app)


def test_register_user_success():
    """Test successful user registration"""
    response = client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'username' in data
    assert data['username'] == 'test'


def test_register_user_duplicate_email():
    """Test registration with duplicate email"""
    # First registration
    client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    # Try to register again with same email
    response = client.post('/api/register', json={
        'name': 'Another User',
        'email': 'test@example.com',
        'password': 'password456'
    })
    
    assert response.status_code == 400
    assert 'already registered' in response.json()['detail'].lower()


def test_register_user_short_password():
    """Test registration with short password"""
    response = client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test@example.com',
        'password': '12345'
    })
    
    assert response.status_code == 400
    assert 'at least 6 characters' in response.json()['detail'].lower()


def test_register_user_invalid_email():
    """Test registration with invalid email format"""
    response = client.post('/api/register', json={
        'name': 'Test User',
        'email': 'invalid-email',
        'password': 'password123'
    })
    
    assert response.status_code == 400
    assert 'invalid email' in response.json()['detail'].lower()


def test_verify_email_success(setup_db):
    """Test successful email verification"""
    engine = setup_db
    # Register user
    response = client.post('/api/register', json={
        'name': 'Test User',
        'email': 'verify@example.com',
        'password': 'password123'
    })
    assert response.status_code == 200
    
    # Get verification token from database
    with Session(engine) as db:
        user = db.exec(select(User).where(User.email == 'verify@example.com')).first()
        assert user is not None
        assert user.email_verified is False
        token = user.verification_token
    
    # Verify email
    response = client.post('/api/verify-email', json={'token': token})
    assert response.status_code == 200
    data = response.json()
    assert 'verified successfully' in data['message'].lower()
    
    # Check user is verified in database
    with Session(engine) as db:
        user = db.exec(select(User).where(User.email == 'verify@example.com')).first()
        assert user.email_verified is True
        assert user.verification_token is None


def test_verify_email_invalid_token():
    """Test email verification with invalid token"""
    response = client.post('/api/verify-email', json={'token': 'invalid-token-12345'})
    
    assert response.status_code == 400
    assert 'invalid' in response.json()['detail'].lower()


def test_resend_verification_success():
    """Test resending verification email"""
    # Register user
    client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    # Resend verification
    response = client.post('/api/resend-verification?email=test@example.com')
    
    assert response.status_code == 200
    data = response.json()
    assert 'sent successfully' in data['message'].lower()


def test_resend_verification_nonexistent_user():
    """Test resending verification for non-existent user"""
    response = client.post('/api/resend-verification?email=nonexistent@example.com')
    
    assert response.status_code == 404
    assert 'not found' in response.json()['detail'].lower()


def test_resend_verification_already_verified(setup_db):
    """Test resending verification for already verified user"""
    engine = setup_db
    # Register and verify user
    client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    # Get token and verify
    with Session(engine) as db:
        user = db.exec(select(User).where(User.email == 'test@example.com')).first()
        token = user.verification_token
    
    client.post('/api/verify-email', json={'token': token})
    
    # Try to resend
    response = client.post('/api/resend-verification?email=test@example.com')
    
    assert response.status_code == 400
    assert 'already verified' in response.json()['detail'].lower()


def test_user_data_stored_correctly(setup_db):
    """Test that user data is stored correctly in database"""
    engine = setup_db
    response = client.post('/api/register', json={
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'password': 'securepass123'
    })
    
    assert response.status_code == 200
    
    # Check database
    with Session(engine) as db:
        user = db.exec(select(User).where(User.email == 'john.doe@example.com')).first()
        assert user is not None
        assert user.name == 'John Doe'
        assert user.email == 'john.doe@example.com'
        assert user.username == 'john.doe'
        assert user.email_verified is False
        assert user.verification_token is not None
        assert user.verification_token_expiry is not None
