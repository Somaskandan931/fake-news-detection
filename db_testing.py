from pymongo import MongoClient

# Try connecting to the existing database
try:
    client = MongoClient("mongodb://localhost:27017/")  # Change this if needed
    db_list = client.list_database_names()

    if "admin" in db_list:
        print("✅ Connected to Local MongoDB")
    else:
        print("✅ Connected to MongoDB Atlas or another remote server")

    print(f"Databases available: {db_list}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
