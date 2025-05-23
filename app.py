import warnings
import os
import json
import asyncio
import aiohttp
import time
import sqlite3
from functools import lru_cache
from PIL import Image
import torch
from flask import Flask, request, render_template, Response, jsonify
from bs4 import BeautifulSoup
from transformers import BlipProcessor, BlipForConditionalGeneration
from werkzeug.utils import secure_filename
from html import escape
import concurrent.futures
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

CONFIG = {
    'UPLOAD_FOLDER': 'static/uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'HTTP_TIMEOUT': 5,  # Timeout for HTTP requests
    'MAX_RESULTS': 20,  # Reduced for faster response
    'CACHE_SIZE': 10,  # Max cached queries (for in-memory fallback)
    'FILE_DELETE_RETRIES': 3,  # Number of retries for file deletion
    'FILE_DELETE_DELAY': 0.5,  # Delay between retries in seconds
    'DB_PATH': 'products.db',  # SQLite database path
    'CACHE_TTL': 24 * 60 * 60  # Cache TTL in seconds (24 hours)
}

app.config['UPLOAD_FOLDER'] = CONFIG['UPLOAD_FOLDER']
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']
os.makedirs(CONFIG['UPLOAD_FOLDER'], exist_ok=True)

# Thread pool for BLIP captioning
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Predefined categories for default display
DEFAULT_CATEGORIES = ["t-shirt", "Shoes"]

# Load BLIP model
logging.info("⚙️ Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
logging.info("✅ Captioning model loaded successfully.")

# SQLite Database Setup
def init_db():
    """Initialize SQLite database and create search_results table."""
    with sqlite3.connect(CONFIG['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                min_price REAL,
                max_price REAL,
                source TEXT,
                title TEXT,
                price REAL,
                rating TEXT,
                image TEXT,
                link TEXT,
                category TEXT,
                timestamp INTEGER
            )
        ''')
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query ON search_results (query, min_price, max_price)')
        conn.commit()
    logging.info("✅ SQLite database initialized.")

init_db()

# Utility Functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG['ALLOWED_EXTENSIONS']

async def get_image_caption(image_path):
    """Generate a caption for the given image using BLIP."""
    try:
        logging.info(f"📸 Opening image: {image_path}")
        loop = asyncio.get_running_loop()
        def open_image():
            with Image.open(image_path) as img:
                return img.convert('RGB').copy()
        raw_image = await loop.run_in_executor(None, open_image)
        logging.info("🧠 Generating caption...")
        inputs = await loop.run_in_executor(None, lambda: processor(raw_image, return_tensors="pt"))
        out = await loop.run_in_executor(None, lambda: model.generate(**inputs))
        caption = processor.decode(out[0], skip_special_tokens=True)
        logging.info(f"📝 Caption generated: {caption}")
        return caption
    except Exception as e:
        logging.error(f"❌ Caption Error: {e}")
        return None

def safe_remove_file(filepath):
    """Attempt to remove a file with retries to handle Windows file locking."""
    for attempt in range(CONFIG['FILE_DELETE_RETRIES']):
        try:
            os.remove(filepath)
            logging.info(f"🗑️ File deleted: {filepath}")
            return
        except PermissionError as e:
            logging.warning(f"⚠️ File deletion attempt {attempt + 1} failed: {e}")
            time.sleep(CONFIG['FILE_DELETE_DELAY'])
        except Exception as e:
            logging.error(f"❌ File deletion error: {e}")
            return
    logging.error(f"❌ Failed to delete file after {CONFIG['FILE_DELETE_RETRIES']} attempts: {filepath}")

def save_to_db(query, min_price, max_price, results):
    """Save search results to SQLite database."""
    try:
        with sqlite3.connect(CONFIG['DB_PATH']) as conn:
            cursor = conn.cursor()
            timestamp = int(time.time())
            for result in results:
                cursor.execute('''
                    INSERT INTO search_results (
                        query, min_price, max_price, source, title, price, rating, image, link, category, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query, min_price, max_price, result['source'], result['title'], result['price'],
                    result['rating'], result['image'], result['link'], result['category'], timestamp
                ))
            conn.commit()
            logging.info(f"💾 Saved {len(results)} results to database for query: {query}")
    except Exception as e:
        logging.error(f"❌ Database save error: {e}")

def get_from_db(query, min_price, max_price):
    """Fetch cached results from SQLite database."""
    try:
        with sqlite3.connect(CONFIG['DB_PATH']) as conn:
            cursor = conn.cursor()
            # Fetch results within TTL and matching query/price filters
            cursor.execute('''
                SELECT source, title, price, rating, image, link, category
                FROM search_results
                WHERE query = ? AND
                      (min_price IS ? OR min_price = ?) AND
                      (max_price IS ? OR max_price = ?) AND
                      timestamp > ?
                LIMIT ?
            ''', (
                query, min_price, min_price, max_price, max_price,
                int(time.time()) - CONFIG['CACHE_TTL'], CONFIG['MAX_RESULTS']
            ))
            rows = cursor.fetchall()
            results = [{
                'source': row[0], 'title': row[1], 'price': row[2], 'rating': row[3],
                'image': row[4], 'link': row[5], 'category': row[6]
            } for row in rows]
            logging.info(f"📦 Fetched {len(results)} cached results for query: {query}")
            return results
    except Exception as e:
        logging.error(f"❌ Database fetch error: {e}")
        return []

def clean_db():
    """Remove expired cache entries from database."""
    try:
        with sqlite3.connect(CONFIG['DB_PATH']) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM search_results
                WHERE timestamp < ?
            ''', (int(time.time()) - CONFIG['CACHE_TTL'],))
            conn.commit()
            logging.info(f"🧹 Cleaned {cursor.rowcount} expired cache entries.")
    except Exception as e:
        logging.error(f"❌ Database cleanup error: {e}")

# Search Logic
async def search_amazon(query, min_price=None, max_price=None):
    """Search Amazon for products matching the query with price filtering."""
    # Check database cache first
    cached_results = get_from_db(query, min_price, max_price)
    if cached_results:
        return cached_results

    logging.info(f"🔍 Searching Amazon for: '{query}'")
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG['HTTP_TIMEOUT'])) as session:
        try:
            url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"❌ Amazon HTTP Error: {response.status}")
                    return results
                soup = BeautifulSoup(await response.text(), 'html.parser')
                items = soup.select('div.s-result-item[data-component-type="s-search-result"]')[:CONFIG['MAX_RESULTS']]
                for item in items:
                    try:
                        title = item.select_one('h2 span').text[:100] if item.select_one('h2 span') else ''
                        price_whole = item.select_one('span.a-price-whole').text.replace(',', '') if item.select_one('span.a-price-whole') else '0'
                        price = float(price_whole) if price_whole else 0.0
                        rating = item.select_one('span.a-icon-alt').text.split(' ')[0] if item.select_one('span.a-icon-alt') else 'N/A'
                        image = item.select_one('img.s-image')['src'] if item.select_one('img.s-image') else ''
                        link = item.select_one('a.a-link-normal.s-no-outline')['href'] if item.select_one('a.a-link-normal.s-no-outline') else '#'
                        if (min_price is None or price >= min_price) and (max_price is None or price <= max_price):
                            results.append({
                                "source": "Amazon",
                                "title": title,
                                "price": price,
                                "rating": rating,
                                "image": image,
                                "link": f"https://www.amazon.in{link}",
                                "category": query
                            })
                    except Exception as e:
                        logging.debug(f"Amazon item parse error: {e}")
                        continue
                logging.info(f"✅ Found {len(results)} products on Amazon.")
                if results:
                    save_to_db(query, min_price, max_price, results)
        except Exception as e:
            logging.error(f"❌ Amazon Error: {e}")
    return results


async def collect_results(query, min_price, max_price):
    """Collect search results from Amazon and Flipkart."""
    tasks = []
    if query:
        tasks.extend([
            search_amazon(query, min_price, max_price),
            # search_flipkart(query, min_price, max_price)
        ])
    else:
        for category in DEFAULT_CATEGORIES:
            tasks.extend([
                search_amazon(category, min_price, max_price),
                # search_flipkart(category, min_price, max_price)
            ])
    results = await asyncio.gather(*tasks, return_exceptions=True)
    collected = []
    for result in results:
        if not isinstance(result, Exception):
            collected.append(result)
        else:
            collected.append({"error": str(result)})
    # Clean database periodically (e.g., every 10 requests)
    if len(tasks) % 10 == 0:
        clean_db()
    return collected

# Routes
@app.route('/')
def index():
    """Render the main search page with initial data."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with text, image, or default random products."""
    query = None
    min_price = request.form.get('min_price', type=float)
    max_price = request.form.get('max_price', type=float)
    generated_caption = None

    # Validate price inputs
    if min_price is not None and min_price < 0:
        return jsonify({"error": "Minimum price cannot be negative"}), 400
    if max_price is not None and max_price < 0:
        return jsonify({"error": "Maximum price cannot be negative"}), 400
    if min_price is not None and max_price is not None and min_price > max_price:
        return jsonify({"error": "Minimum price cannot exceed maximum price"}), 400

    # Handle text query
    if 'query' in request.form and request.form['query'].strip():
        query = escape(request.form['query'].strip())

    # Handle image upload
    filepath = None
    if 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                generated_caption = loop.run_until_complete(get_image_caption(filepath))
            finally:
                loop.close()
            if generated_caption:
                query = generated_caption
                stop_words = ["a", "man", "woman", "wearing", "is", "in", "the", "with"]
                query = " ".join([w for w in query.split() if w.lower() not in stop_words])

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(collect_results(query, min_price, max_price))
        finally:
            loop.close()

        def generate():
            for result in results:
                if generated_caption and result:
                    yield json.dumps({"caption": generated_caption, "results": result}) + '\n'
                elif not query and result:
                    yield json.dumps({"caption": "Explore Popular Products", "results": result}) + '\n'
                else:
                    yield json.dumps(result) + '\n'
        return Response(generate(), mimetype='application/json')
    finally:
        if filepath and os.path.exists(filepath):
            safe_remove_file(filepath)

# if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=10000)
app.run(debug=True)
