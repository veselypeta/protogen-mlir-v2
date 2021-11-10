set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPCCPassesIncGen)
add_dependencies(mlir-headers MLIRPCCPassesIncGen)

# Patters
set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(Patterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPCCPatternsIncGen)
add_dependencies(mlir-headers MLIRPCCPatternsIncGen)