#pragma once
#include "ProtoCCParser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace mlir {
class MLIRContext;
class OwningModuleRef;

namespace pcc {
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx,
                       std::string filename = "");

} // namespace pcc

namespace fsm {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx,
                       std::string filename = "");
}
} // namespace mlir
