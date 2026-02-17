import sys
import os

# Add parent directory to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.open_alex_tool import search_openalex

def test_search():
    print("Testing OpenAlex Search Tool...")
    query = "machine learning"
    try:
        result = search_openalex(query)
        print("Search successful!")
        print("Result preview:")
        print(result[:500] + "..." if len(result) > 500 else result)
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_search()
