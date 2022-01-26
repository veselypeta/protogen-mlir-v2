#pragma once

#include "FSM/FSMOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace fsm {

#define GEN_PASS_CLASSES
#include "FSM/Passes/Passes.h.inc"

void registerFullOptimizationPipeline();

} // namespace pcc
} // namespace mlir


