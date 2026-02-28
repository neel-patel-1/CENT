import torch
from cent_simulation.aim_sim import PIM
from cent_simulation.utils import get_args

# Initialize PIM simulator
args = get_args()
pim = PIM(args)

# Example 1: Single bank write and read
print("=== Test 1: Single Bank Write/Read ===")

# Create test data (must fit in a row - default DRAM_column=1024)
test_data = torch.randn(512)  # 512 elements
print(f"Original data shape: {test_data.shape}")

# Store to DRAM: (dimm_index, channel_index, bank_index, row_index, col_index, size, data, trace_flag)
dimm = 0
channel = 0
bank = 0
row = 0
col = 0
pim.store_to_DRAM_single_bank(dimm, channel, bank, row, col, len(test_data), test_data, op_trace=False)

# Update timing manually (simulator doesn't auto-track single bank ops in this low-level API)
pim.time["WR_SBK"] += pim.timing_constant["WR_SBK"] + len(test_data) // pim.burst_length

# Load from DRAM
loaded_data = pim.load_from_DRAM_single_bank(dimm, channel, bank, row, col, len(test_data), op_trace=False)
pim.time["RD_SBK"] += pim.timing_constant["RD_SBK"] + len(test_data) // pim.burst_length

# Verify correctness
if torch.equal(test_data, loaded_data):
    print("✓ Data verified: Write/Read successful!")
else:
    print("✗ Data mismatch!")
    max_error = torch.max(torch.abs(test_data - loaded_data))
    print(f"  Max error: {max_error}")

# Show performance
print(f"Write cycles: {pim.timing_constant['WR_SBK'] + len(test_data) // pim.burst_length}")
print(f"Read cycles: {pim.timing_constant['RD_SBK'] + len(test_data) // pim.burst_length}")
print(f"Total cycles: {pim.time['WR_SBK'] + pim.time['RD_SBK']}")


# Example 2: Multi-bank operations with timing
print("\n=== Test 2: Multi-Bank Write/Read ===")

# Reset timing for clean measurement
pim.time["WR_SBK"] = 0
pim.time["RD_SBK"] = 0

# Create larger dataset distributed across multiple banks
num_banks = 4
data_per_bank = 256
test_data_multi = [torch.randn(data_per_bank) for _ in range(num_banks)]

# Write to multiple banks
for bank_idx in range(num_banks):
    pim.store_to_DRAM_single_bank(0, 0, bank_idx, 0, 0, data_per_bank, test_data_multi[bank_idx], False)
    pim.time["WR_SBK"] += pim.timing_constant["WR_SBK"] + data_per_bank // pim.burst_length

# Read back from multiple banks
loaded_data_multi = []
for bank_idx in range(num_banks):
    data = pim.load_from_DRAM_single_bank(0, 0, bank_idx, 0, 0, data_per_bank, False)
    loaded_data_multi.append(data)
    pim.time["RD_SBK"] += pim.timing_constant["RD_SBK"] + data_per_bank // pim.burst_length

# Verify all banks
all_correct = True
for bank_idx in range(num_banks):
    if not torch.equal(test_data_multi[bank_idx], loaded_data_multi[bank_idx]):
        print(f"✗ Bank {bank_idx} data mismatch!")
        all_correct = False

if all_correct:
    print(f"✓ All {num_banks} banks verified successfully!")

print(f"Total write cycles: {pim.time['WR_SBK']:.1f}")
print(f"Total read cycles: {pim.time['RD_SBK']:.1f}")
print(f"Combined cycles: {pim.time['WR_SBK'] + pim.time['RD_SBK']:.1f}")


# Example 3: Using PIM operations (MAC, EWMUL) with verification
print("\n=== Test 3: PIM MAC Operation ===")

# Reset timing
pim.time["MAC_BK_GB"] = 0
pim.time["WR_GB"] = 0
pim.time["RD_MAC"] = 0

# Store data in bank and global buffer
vec_a = torch.ones(pim.burst_length) * 2.0  # Store in bank
vec_b = torch.ones(pim.burst_length) * 3.0  # Store in GB

pim.store_to_DRAM_single_bank(0, 0, 0, 0, 0, pim.burst_length, vec_a, False)
pim.WR_GB(0, 0, 1, 0, 1, vec_b, False)  # op_size=1 means 1 burst

# Initialize MAC register
pim.WR_BIAS(0, 0, 1, 0, [0.0] * pim.num_banks, False)

# Perform MAC: accumulate dot product
pim.MAC_BK_GB(0, 0, 1, 0, 0, 0, 1, False, "MAC_BK_GB")  # row=0, col=0, latch=0, op_size=1

# Read result
result = pim.RD_MAC(0, 0, 1, 0, False)

# Expected: sum(2.0 * 3.0 for 16 elements) = 96.0
expected = (vec_a * vec_b).sum()
if torch.isclose(torch.tensor(result[0]), expected):
    print(f"✓ MAC verified: {result[0]:.1f} == {expected:.1f}")
else:
    print(f"✗ MAC mismatch: {result[0]:.1f} vs expected {expected:.1f}")

print(f"MAC operation cycles: {pim.time['MAC_BK_GB']:.1f}")


# Summary of all timing
print("\n=== Complete Performance Summary ===")
print(f"Timing constants (base costs):")
print(f"  WR_SBK base: {pim.timing_constant['WR_SBK']} cycles")
print(f"  RD_SBK base: {pim.timing_constant['RD_SBK']} cycles")
print(f"  MAC_BK_GB base: {pim.timing_constant['MAC_BK_GB']} cycles")
print(f"\nActual operation costs:")
for op_type, cycles in pim.time.items():
    if cycles > 0:
        print(f"  {op_type}: {cycles:.1f} cycles")

print(f"\nTotal simulated cycles: {sum(pim.time.values()):.1f}")
print(f"Time at 2 GHz: {sum(pim.time.values()) * 0.5:.1f} ns")

# Close trace file
pim.file.close()
print("\n✓ Test complete!")