from app import app, db, Authenticate, bcrypt

# Create an application context
with app.app_context():
    # Create the database and the db table
    db.create_all()

    # Insert user data
    hashed_password1 = bcrypt.generate_password_hash('devtest123').decode('utf-8')
    user1 = Authenticate(username='devtest', password=hashed_password1)

    hashed_password2 = bcrypt.generate_password_hash('test').decode('utf-8')
    user2 = Authenticate(username='test', password=hashed_password2)

    db.session.add(user1)
    db.session.add(user2)

    # commit the changes
    db.session.commit()