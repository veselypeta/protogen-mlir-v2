#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "ProtoCCParser.h"

namespace mlir
{
    class MLIRContext;
    class OwningModuleRef;
} // namespace mlir

namespace pcc
{
    /// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
    /// or nullptr on failure.
    mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx, ProtoCCParser::DocumentContext *docCtx);

} // namespace pcc
