# Import the Flask app from webapp/app.py
import sys
import os
# Add the project root (parent of api/) so 'webapp' package can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.app import app

# This is the entry point for Vercel
if __name__ == "__main__":
    app.run()
