# setup the passes
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRFSMPassesIncGen)
add_dependencies(mlir-headers MLIRFSMPassesIncGen)