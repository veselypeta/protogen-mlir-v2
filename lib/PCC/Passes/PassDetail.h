#pragma once

#include "PCC/PCCOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pcc {

void registerStableCompilation();

#define GEN_PASS_CLASSES
#include "PCC/Passes/Passes.h.inc"
} // namespace pcc
} // namespace mlir
