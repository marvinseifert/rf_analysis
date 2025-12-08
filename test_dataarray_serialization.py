"""
Test script to verify xarray DataArray serialization/deserialization with Pydantic
"""
from pathlib import Path
import xarray as xr
from pydantic import BaseModel
from organize.configs import SerializedDataArray


# Create a simple test model
class TestModel(BaseModel):
    data: SerializedDataArray
    name: str


# Test 1: Create a DataArray
print("Test 1: Creating a model with xr.DataArray...")
test_data = xr.DataArray([1, 2, 3, 4], dims=["x"], coords={"x": [10, 20, 30, 40]})
model = TestModel(data=test_data, name="test")
print(f"✓ Model created: data type = {type(model.data)}")
print(f"  DataArray: {model.data.values}")

# Test 2: Serialize to JSON
print("\nTest 2: Serializing to JSON...")
json_str = model.model_dump_json(indent=2)
print(f"✓ JSON serialized (first 200 chars):\n{json_str[:200]}...")

# Test 3: Deserialize from JSON
print("\nTest 3: Deserializing from JSON...")
model_loaded = TestModel.model_validate_json(json_str)
print(f"✓ Model loaded: data type = {type(model_loaded.data)}")
print(f"  DataArray: {model_loaded.data.values}")

# Test 4: Verify data integrity
print("\nTest 4: Verifying data integrity...")
assert isinstance(
    model_loaded.data, xr.DataArray
), f"Expected xr.DataArray, got {type(model_loaded.data)}"
assert (model_loaded.data.values == test_data.values).all(), "Data values don't match!"
assert model_loaded.name == "test", "Name doesn't match!"
print("✓ All checks passed! Data is correctly deserialized as xr.DataArray")

# Test 5: Test with None value
print("\nTest 5: Testing with None value...")
model_none = TestModel(data=None, name="none_test")
json_none = model_none.model_dump_json()
model_none_loaded = TestModel.model_validate_json(json_none)
assert model_none_loaded.data is None, "None value not preserved!"
print("✓ None value handled correctly")

print("\n" + "=" * 50)
print("ALL TESTS PASSED! ✓")
print("=" * 50)
