#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "FSM/FSMDialect.h"
#include "FSM/FSMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "FSM/FSM.h.inc"
