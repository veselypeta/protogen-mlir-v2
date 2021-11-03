#include "PCC/Passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "PassDetail.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "PCC/Passes/Passes.h.inc"
} // namespace


void mlir::pcc::initAllPCCPasses(){
  registerPasses();
}

void mlir::pcc::initAllPipelines(){
  mlir::pcc::registerStableCompilation();
}

