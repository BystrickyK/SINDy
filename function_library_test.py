from function_library import *

sys = LorenzSystem([10, 12, 14])
sys.propagate(300)
# sim_data = np.array([[0,1,2,3,4,5], [1,2,3,4,5,6], [1,3,5,7,9,11], [-1,-2,-3,-4,-5,-6]]).T
sig = Signal(sys.sim_data)

# polys = sym_function_library(sig, [1,2])
poly_lib = function_library(sig.x, [*range(1,5)])

print(poly_lib.head())
print(poly_lib.columns)
