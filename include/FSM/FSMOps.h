#pragma once
#include "FSM/FSMDialect.h"
#include "FSM/FSMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace fsm {
namespace impl {
LogicalResult verifySameTypeOperands(Operation *op);
}
} // namespace fsm
} // namespace mlir

#define GET_OP_CLASSES
#include "FSM/FSM.h.inc"
