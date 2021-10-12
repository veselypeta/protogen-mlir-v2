add_protogen_dialect(PCC pcc)

set(LLVM_TARGET_DEFINITIONS PCC.td)

mlir_tablegen(PCCAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PCCAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPCCAttrIncGen)
add_dependencies(mlir-headers MLIRPCCAttrIncGen)
