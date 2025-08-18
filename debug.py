import asyncio
import logging
from Source.Services.search_on_index import AdvancedUnifiedContentSearch

# Setup basic logging
logging.basicConfig(level=logging.INFO)


async def debug_index_fields():
    """Debug function to check what fields exist in the index"""
    print("🔍 Debugging index fields...")

    try:
        # Create search instance
        search = AdvancedUnifiedContentSearch()

        # Get a sample document with all fields
        print("\n📋 Getting sample document with all fields...")
        results = search.search_client.search(
            search_text="*",
            select=["*"],  # Select all fields
            top=1
        )

        docs = list(results)
        if docs:
            sample_doc = docs[0]
            print(f"\n✅ Found sample document with ID: {sample_doc.get('id', 'N/A')}")
            print("\n📝 Available fields in index:")
            print("-" * 50)

            for key, value in sample_doc.items():
                if key.startswith('@'):
                    continue  # Skip Azure Search metadata fields
                print(f"  🔹 {key}: {type(value).__name__}")
                if isinstance(value, str) and len(value) > 100:
                    print(f"      Preview: {value[:100]}...")
                else:
                    print(f"      Value: {value}")
                print()

            print("-" * 50)
            print(f"📊 Total fields found: {len([k for k in sample_doc.keys() if not k.startswith('@')])}")

            # Check specifically for file_name field
            if 'file_name' in sample_doc:
                print(f"✅ file_name field EXISTS: {sample_doc['file_name']}")
            else:
                print("❌ file_name field NOT FOUND")

                # Look for similar fields
                similar_fields = [k for k in sample_doc.keys() if 'file' in k.lower() or 'name' in k.lower()]
                if similar_fields:
                    print(f"🔍 Similar fields found: {similar_fields}")
        else:
            print("❌ No documents found in index")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_index_fields())
