import ee

try:
    # Try to initialize without project (will show current project)
    ee.Initialize()
    print("✅ Earth Engine initialized successfully!")
    
    # Try to get project info
    try:
        # This might work to show current project
        print("Current project info:")
        print(f"EE URL: {ee.data.getInfo()}")
    except:
        pass
    
    # Test with a simple operation
    test = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318')
    print("✅ Can access Earth Engine data")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("You may need to re-authenticate or set a project")
 